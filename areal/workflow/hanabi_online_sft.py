import asyncio
import json
import os
import time
import uuid
from typing import Optional

import aiofiles
import torch

from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from areal.workflow.hanabi import HanabiWorkflow, _sanitize_for_json

from areal.utils import logging

logger = logging.getLogger("Hanabi Online SFT Workflow")

class HanabiOnlineSFTWorkflow(HanabiWorkflow, RolloutWorkflow):
    """Hanabi workflow specialized for online SFT without teacher forcing.

    This workflow reuses the core rollout logic from :class:`HanabiWorkflow`
    while disabling teacher interactions and filtering trajectories based on a
    running mean score threshold.
    """

    def __init__(
        self,
        *args,
        score_multiplier: float = 1.0,
        env_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            env_kwargs=env_kwargs,
            teacher_rollout=None,
            teacher_tokenizer=None,
            teacher_api_key=None,
            teacher_api_model=None,
            sft_reg=0.0,
            **kwargs,
        )
        self.score_multiplier = score_multiplier
        self._running_score_sum = 0.0
        self._running_score_count = 0

    async def arun_episode(self, engine, data):
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
        ]
        episodes = await asyncio.gather(*tasks)

        accepted = []
        accepted_metadata = []
        while len(accepted) < self.gconfig.n_samples:
            threshold = None
            for episode in episodes:
                (
                    res_list,
                    prompt_strs,
                    completions_strs,
                    rewards,
                    seqlens,
                    trajectory,
                    qa_logs,
                    teacher_logs,
                    step_logs,
                    episode_info,
                ) = episode

                final_score = episode_info.get(
                    "final_score", episode_info.get("total_reward", 0.0)
                )
                threshold = None
                if self._running_score_count > 0:
                    mean_score = self._running_score_sum / self._running_score_count
                    threshold = mean_score * self.score_multiplier

                self._running_score_sum += float(final_score)
                self._running_score_count += 1

                if threshold is not None and final_score < threshold:
                    continue

                for td in res_list:
                    td["advantages"] = torch.ones_like(td["rewards"])
                    td["rewards"] = torch.ones_like(td["rewards"])
                accepted.extend(res_list)
                accepted_metadata.append(
                    (
                        prompt_strs,
                        completions_strs,
                        rewards,
                        seqlens,
                        trajectory,
                        qa_logs,
                        teacher_logs,
                        step_logs,
                        {**episode_info, "threshold": threshold},
                    )
                )

            logger.info(f"Running mean score: {self._running_score_sum / self._running_score_count}, threshold: {threshold}, accpeted: {len(accepted)} / {self.gconfig.n_samples}, {'continuing generation...' if len(accepted) < self.gconfig.n_samples else 'stopping and preparing data batch'}")
            if len(accepted) < self.gconfig.n_samples:
                tasks = [
                    self._run_one_episode(engine, data, rid)
                ]
                episodes = await asyncio.gather(*tasks)
            # elif len(accepted) > self.gconfig.n_samples:
            #     accepted = accepted[: self.gconfig.n_samples]
            #     accepted_metadata = accepted_metadata[: self.gconfig.n_samples]
            #     break
            # else:
            #     break

        if self.dump_dir is not None and accepted_metadata:
            version = engine.get_version()
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            file_path = os.path.join(dump_path, f"{qid}.jsonl")
            async with aiofiles.open(file_path, "a") as f:
                for episode_idx, meta in enumerate(accepted_metadata):
                    (
                        p_list,
                        c_list,
                        r_list,
                        sl_list,
                        traj,
                        qa_logs,
                        t_logs,
                        step_logs,
                        episode_info,
                    ) = meta
                    record = {
                        "query_id": qid,
                        "episode_index": episode_idx,
                        "prompts": p_list,
                        "completions": c_list,
                        "sequence_lengths": sl_list,
                        "returns": r_list,
                        "trajectory": traj,
                        "qa_logs": qa_logs,
                        "teacher_logs": t_logs,
                        "steps": step_logs,
                        "summary": {
                            **episode_info,
                            "model_version": version,
                        },
                        "threshold_multiplier": self.score_multiplier,
                        "timestamp": time.time(),
                    }
                    await f.write(json.dumps(_sanitize_for_json(record)) + "\n")

        return concat_padded_tensors(accepted)
