import asyncio
from sys import exc_info
import aiohttp
import json
from tenacity import retry, stop_after_attempt, wait_exponential

OPENAI_API_KEY = "sk-43dd5f664179406d92fec42a9364f8a5"
OPENAI_API_BASE = "https://matrixllm.alipay.com/v1"

LAST_USAGE = None
USAGE_TEMPLATE = dict(
    prompt_tokens=0,
    completion_tokens=0,
    reasoning_tokens=0,
)
USAGE = dict()
ACTUAL_USAGE = dict()

# Add asyncio lock for thread-safe usage updates
_usage_lock = asyncio.Lock()

def load_usage(usg):
    global USAGE, ACTUAL_USAGE
    USAGE = usg["usage"]
    ACTUAL_USAGE = usg["actual_usage"]

def copy_usage():
    global USAGE, ACTUAL_USAGE
    return dict(usage=USAGE.copy(), actual_usage=ACTUAL_USAGE.copy())

async def copy_usage_async():
    """Thread-safe async version of copy_usage() for use in coroutines."""
    global USAGE, ACTUAL_USAGE, _usage_lock
    async with _usage_lock:
        return dict(usage=USAGE.copy(), actual_usage=ACTUAL_USAGE.copy())

def apply_usage():
    global USAGE, ACTUAL_USAGE, LAST_USAGE
    if LAST_USAGE is not None:
        model = LAST_USAGE.get("model", "unkown")
        if model not in ACTUAL_USAGE:
            ACTUAL_USAGE[model] = USAGE_TEMPLATE.copy()
        ACTUAL_USAGE[model]["prompt_tokens"] += LAST_USAGE.get("prompt_tokens", 0)
        ACTUAL_USAGE[model]["completion_tokens"] += LAST_USAGE.get("completion_tokens", 0)
        ACTUAL_USAGE[model]["reasoning_tokens"] += LAST_USAGE.get("reasoning_tokens", 0)

async def apply_usage_async():
    """Thread-safe async version of apply_usage() for use in coroutines."""
    global USAGE, ACTUAL_USAGE, LAST_USAGE, _usage_lock
    async with _usage_lock:
        if LAST_USAGE is not None:
            model = LAST_USAGE.get("model", "unkown")
            if model not in ACTUAL_USAGE:
                ACTUAL_USAGE[model] = USAGE_TEMPLATE.copy()
            ACTUAL_USAGE[model]["prompt_tokens"] += LAST_USAGE.get("prompt_tokens", 0)
            ACTUAL_USAGE[model]["completion_tokens"] += LAST_USAGE.get("completion_tokens", 0)
            ACTUAL_USAGE[model]["reasoning_tokens"] += LAST_USAGE.get("reasoning_tokens", 0)

async def handle_streaming_response(response, model_name, return_raw_result=False):
    """
    Handle streaming response from the LLM API.
    Yields chunks of data as they arrive.
    """
    full_content = ""
    reasoning_content = ""
    full_response = {
        "id": None,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
            },
            "finish_reason": None
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    
    async for line in response.content:
        line = line.decode('utf-8').strip()
        
        # Skip empty lines and the "data: [DONE]" message
        # if line == "data:[DONE]":
        #     print(line, flush=True)
        if not line or line == "data:[DONE]":
            continue
        
        # Parse SSE format
        if line.startswith("data:"):
            line = line[5:]  # Remove "data: " prefix
        
        try:
            data = json.loads(line) 

            # print(line, flush=True)
            
            # Update response metadata
            if full_response["id"] is None and "id" in data:
                full_response["id"] = data["id"]
            
            # Handle choices
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                
                # Handle delta content
                if "delta" in choice:
                    content_chunk = choice["delta"].get("content", None)
                    reasoning_chunk = choice["delta"].get("reasoning_content", None)
                    if content_chunk:
                        full_content += content_chunk
                        full_response["choices"][0]["message"]["content"] = full_content
                        
                    if reasoning_chunk:
                        # print("[DEBUG] reasoning content detected", flush=True)
                        reasoning_content += reasoning_chunk
                        full_response["choices"][0]["message"]["reasoning_content"] = reasoning_content
                
                # Update finish reason
                if choice.get("finish_reason"):
                    full_response["choices"][0]["finish_reason"] = choice["finish_reason"]
            
            # Handle usage
            if "usage" in data and data["usage"]:
                # Accumulate usage stats
                full_response["usage"] = data["usage"]

                print(full_response["usage"])
                
                # Update global usage stats
                global USAGE, LAST_USAGE
                if model_name not in USAGE:
                    USAGE[model_name] = USAGE_TEMPLATE.copy()
                
                USAGE[model_name]["prompt_tokens"] += data["usage"].get("prompt_tokens", 0)
                USAGE[model_name]["completion_tokens"] += data["usage"].get("completion_tokens", 0)
                USAGE[model_name]["reasoning_tokens"] += (data["usage"].get("completion_tokens_details", {}) or dict()).get("reasoning_tokens", 0)
                
                LAST_USAGE = data["usage"].copy()
                LAST_USAGE["model"] = model_name
                
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse SSE data: {line}, error: {e}")
            continue
    
    # After streaming is complete, return the full response if needed
    if return_raw_result:
        # For consistency with non-streaming mode, yield the final complete response
        return full_response
    else:
        return full_response["choices"][0]

@retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=4, max=20))
async def async_request_llm(messages, enable_thinking=True, model_name="gpt-5-2025-08-07", stream=True, return_raw_result=False, **additional_kwargs):
    print("[DEBUG] post request to llm", model_name, enable_thinking, additional_kwargs, flush=True)
    headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
    
    if enable_thinking:    
        kwargs = {
            "max_completion_tokens": 128000,
            # "prompt_cache_retention": "24h",
            "reasoning_effort": "medium",
            # "extra_body": {
            #         "thinking": {
            #             "type": "enabled",
            #             "budget_tokens": 16384
            #         },
            #     },
            # temperature=0.6,
            # max_completion_tokens=32000
        }
        if model_name in ["claude-sonnet-4-5-20250929"]:
            kwargs["reasoning_effort"] = "low"
            kwargs["max_tokens"] = 128000
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 16284,
                }
            }
    else:
        kwargs = {
            "max_completion_tokens": 128000,
            # "max_tokens": 64000,
        }
        if model_name in ["claude-sonnet-4-5-20250929"]:
            kwargs["max_tokens"] = 128000
    
    if model_name in ["gpt-5.1", "gpt-5-mini-2025-08-07"]:
        if "stop" in additional_kwargs:
            _ = additional_kwargs.pop("stop")

    kwargs.update(additional_kwargs)

    print(model_name, kwargs, flush=True)
    
    # print("input:\n", messages, flush=True)

    payload = {
        # "model": "glm-4.6",
        # "model": "kimi-k2-thinking",
        # "model": "claude-sonnet-4-20250514", 
        #  "model": "claude-sonnet-4-5-20250929",
        #  "model": "gpt-5-mini-2025-08-07",
        "model": model_name,
        # "model": "gpt-5.1",
        "messages": messages,
        "stream": stream,
        "stream_options": {"include_usage": True},
        **kwargs
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{OPENAI_API_BASE}/chat/completions", headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=100*60,connect=50*60, sock_read=50*60)) as response:
                if stream:
                    return await handle_streaming_response(response, model_name, return_raw_result)
                else:
                    result = await response.json()
                    print(result)
                    # if "choices" in result and result["choices"][0]["finish_reason"] == "tool_calls":
                    #     return result["choices"][0]["message"]
                    if "usage" in result:
                        global USAGE, LAST_USAGE
                        if model_name not in USAGE:
                            USAGE[model_name] = USAGE_TEMPLATE.copy()
                        USAGE[model_name]["prompt_tokens"] += result["usage"]["prompt_tokens"]
                        USAGE[model_name]["completion_tokens"] += result["usage"]["completion_tokens"]
                        USAGE[model_name]["reasoning_tokens"] += (result["usage"].get("completion_tokens_details", {}) or dict()).get("reasoning_tokens", 0)
                        LAST_USAGE = result["usage"].copy()
                        LAST_USAGE["model"] = model_name
                    if "choices" in result:
                        if return_raw_result:
                            return result
                        return result["choices"][0]
    except Exception as e:
        print(f"⚠️  End detection API error: {e}")
        raise e


def request_llm(messages, **additional_kwargs):
    return asyncio.run(async_request_llm(messages, **additional_kwargs))

if __name__ == "__main__":
    messages = [
        {
            "role": "user",
            "content": "explain RL in two sentences.",
        }
    ]
    print(request_llm(messages, stream=True, return_raw_result=True))