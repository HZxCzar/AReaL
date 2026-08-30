# legacy

Configs that no longer compose. Each one names a parent in its `defaults` list that
was removed by `8a60a2cb`, "Retire the wrappers whose base arms were renamed away",
which deleted the base arms but not these referrers. Launching any of them fails at
startup with a Hydra resolution error, so nothing here is runnable as it stands.

They are kept rather than deleted because each records a setting and its reasoning
against a named run, and that argument outlives the file's ability to load.

This directory is a sibling of `math/`, not a child. Nothing under it composes, so
keeping it outside the config tree proper means a recursive sweep over `math/` --
by a script, a loader, or a person reading the tree -- never has to special-case
these files. Anything retired later belongs outside `math/` for the same reason.

## What is here, and the missing parent

    0818/base/explore.yaml                              noeval
    0818/base/life-clip-higher.yaml                     life
    0818/base/private-visibility.yaml                   full-10turns
    0818/2gpu/life-code.yaml                            base/life-code
    0818/2gpu/full-10turns-info-probe-stratified.yaml   base/full-10turns-info-probe-stratified
    0818/8gpu/full-10turns-info-probe-stratified.yaml   base/full-10turns-info-probe-stratified

The last two arrived by dependency rather than on their own. Both resolved their
direct parent, and both were already unlaunchable through it:

    0818/2gpu/life-clip-higher.yaml     -> base/life-clip-higher, which wants life
    0818/4gpu/private-visibility.yaml   -> base/private-visibility, which wants full-10turns

## Reviving one

Restore the named parent, or point the `defaults` entry at a base arm that exists,
then move the file back under `math/`. The relative paths inside these files still
assume the `math/0818/` layout they were written for, so a revived file needs its
`defaults` paths checked rather than only its parent.
