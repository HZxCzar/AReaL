"""Tests for the CodeAct executor, one per property the probes proved is needed."""

import asyncio
import sys

sys.path.insert(0, "/inspire/qb-ilm/project/qproject-fundationmodel/public/wxxu/TAgent/AReaL.worktrees/dev-two-student")

from examples.tutor.core.code_exec import (  # noqa: E402
    CRASH, OK, TIMEOUT, CodeSession, is_constant_print, is_valid_python,
)

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}" + ("" if ok else f", want {want!r}"))
    if not ok:
        FAILS.append(name)


async def main():
    print("== notebook semantics: a bare trailing expression echoes ==")
    s = CodeSession()
    r = await s.run("x = 6 * 7\nx")
    check("bare expression echoes", r.output, "42")
    check("status ok", r.status, OK)

    print("\n== an explicit print still works, and None does not echo ==")
    s = CodeSession()
    check("print", (await s.run("print(1+1)")).output, "2")
    s2 = CodeSession()
    r = await s2.run("y = 5")          # assignment is not an Expr -> nothing echoes
    check("assignment is silent", r.output, "")
    check("assignment counted silent", r.silent, True)
    s3 = CodeSession()
    r = await s3.run("print(None)\nNone")
    check("None value suppressed", r.output, "None")

    print("\n== namespace persists across turns ==")
    s = CodeSession()
    await s.run("vals = {0,1,4,9}")
    r = await s.run("len(vals)")
    check("later cell sees earlier name", r.output, "4")

    print("\n== a crashed cell does not enter history ==")
    s = CodeSession()
    await s.run("good = 1")
    bad = await s.run("undefined_name_here")
    check("crash reported", bad.status, CRASH)
    check("crash carries the message", "NameError" in bad.output, True)
    r = await s.run("good")
    check("good name survived the crash", r.output, "1")
    check("history kept only the good cell", s.turns_kept, 2)

    print("\n== peek does not mutate the session ==")
    s = CodeSession()
    await s.run("a = 1")
    await s.peek("a = 999\na")
    r = await s.run("a")
    check("peek left the namespace alone", r.output, "1")

    print("\n== prose inside the fence is caught, not executed ==")
    check("prose is invalid", is_valid_python("The output will be 7, which means..."), False)
    check("code is valid", is_valid_python("z = 1\nz"), True)
    s = CodeSession()
    r = await s.run("The output will be 7")
    check("prose crashes rather than passing", r.status, CRASH)

    print("\n== constant-print detection ==")
    check("literal print is constant", is_constant_print('print("110001")'), True)
    check("bare literal is constant", is_constant_print("7"), True)
    check("real arithmetic is not", is_constant_print("print(int('101',2) + 3)"), False)
    check("loop is not", is_constant_print("t=0\nfor i in range(3): t+=i\nt"), False)
    check("import is not", is_constant_print("import math\nmath.sqrt(4)"), False)

    print("\n== stdin is closed, so input() fails fast instead of hanging ==")
    s = CodeSession()
    r = await s.run("x = input('give me a number')\nx")
    check("input() does not hang", r.status, CRASH)

    print("\n== runaway loop hits the wall clock, not the event loop ==")
    s = CodeSession(timeout_s=3.0)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    r = await s.run("while True:\n    pass")
    elapsed = loop.time() - t0
    check("timeout status", r.status, TIMEOUT)
    print(f"  ...  killed after {elapsed:.1f}s (cap 3.0)")
    if elapsed > 8.0:
        FAILS.append("timeout took too long")

    print("\n== concurrency: many sessions at once do not block each other ==")
    t0 = loop.time()
    sessions = [CodeSession() for _ in range(12)]
    outs = await asyncio.gather(*[
        sess.run(f"n = {i}\nn * 2") for i, sess in enumerate(sessions)
    ])
    elapsed = loop.time() - t0
    check("all 12 correct", [o.output for o in outs], [str(i * 2) for i in range(12)])
    print(f"  ...  12 concurrent cells in {elapsed:.1f}s")

    print("\n" + ("ALL PASS" if not FAILS else f"FAILURES: {FAILS}"))
    return 1 if FAILS else 0


sys.exit(asyncio.run(main()))
