# 0808 / decidable

The same three arms as `../`, on a task set where teaching can actually decide
the outcome.

## Why

Measured over 9939 episodes from seven runs, keyed on the problem text rather
than task_id, the parent set spends most of a batch on problems that teach
nothing:

    TRIVIAL     36 problems   student solves it unaided 64% of the time
    EASY         7            one turn is always enough
    DECIDABLE  102            needs several turns, outcome varies
    OTHER       32            the tutor wins almost every time
    HOPELESS     1

Per episode that is 27.4% `pre_solved` -- the student was already right before
the tutor spoke -- plus another 15.6% voided by a leak. Roughly 43% of every
batch is not multi-turn teaching at all, which is both why the online
prompt-vs-baseline curves cannot resolve a 1-2 point effect and why the gradient
for in-context adaptation is so thin.

A problem the tutor always wins and one it never wins are equally useless here:
under a group-relative advantage every rollout in the group scores the same, the
advantage is identically zero, and the step trains on nothing.

## What changed

One line. `student_generalize.path` is both the transfer bank AND the dataset
filter -- `train.py:_filter_generated_generalization_dataset` keeps only rows
whose id appears in it -- so pointing it at the smaller bank filters train and
test together. Everything else is inherited unchanged from `../`.

    parent  250 triples   132 train / 118 test
    here    102 triples    66 train /  36 test

## What this costs

An epoch is 4.1 steps instead of 7.4 at `train_dataset.batch_size: 16` (16
groups x G=8 = 128 episodes, so 16 problems a step). Over 100 steps each train
problem is seen ~24 times rather than ~14. That is the trade: no wasted
episodes, more repetition per problem.

36 test problems is thin for comparing arms. Widening the success window does
not help -- [0.05, 0.95] yields the same 36 -- because the limit is coverage,
not the cut: `debug_trace_every_n_rollouts` is 10 and the eval side dumps far
less than the train side, so only 177 of the parent's 250 problems ever reached
a log. The other 73 are unclassified rather than rejected, and are excluded
here. `analysis/hazard_20260806/classify_rest.py` is measuring them; when it
lands, rebuild with `write_decidable.py` and the test split should grow.

## Caveat on the selection

The buckets were measured on runs of the arms being compared. Selecting problems
where THIS teacher lands mid-range is a mild form of fitting the set to the
model: a different teacher would put a different set mid-range. It biases toward
finding differences between arms that are close to today's policy, and away from
generalising to teachers far from it. Acceptable for an arm comparison, not for
a headline number.
