"""executor tests: the argv-arrays-only contract is structural.

The falsifying inputs live here: a string argv, an empty argv, a
non-str element, an unknown reversibility class, a mutating plan
without a revert, a timeout, a failing postcondition with auto-revert,
and a shell metacharacter that must survive as a LITERAL argument.
"""
import unittest
from unittest import mock

from assistant.executor import ActionPlan, Step, run, run_step
from assistant.executor import runner as runner_mod


class TestStepContract(unittest.TestCase):
    def test_string_argv_is_rejected(self):
        with self.assertRaises(TypeError):
            Step("pacman -Q")  # the classic shell-string mistake

    def test_empty_argv_rejected(self):
        with self.assertRaises(TypeError):
            Step(())

    def test_non_str_element_rejected(self):
        with self.assertRaises(TypeError):
            Step(("echo", 42))

    def test_metacharacters_stay_literal(self):
        # ';' must never be interpreted — it is one argument
        s = Step(("echo", "a; rm -rf /"))
        self.assertEqual(s.argv[1], "a; rm -rf /")


class TestPlanContract(unittest.TestCase):
    def test_unknown_reversibility_rejected(self):
        with self.assertRaises(TypeError):
            ActionPlan(steps=(Step(("true",)),), reversibility="yolo")

    def test_mutating_plan_requires_revert(self):
        with self.assertRaises(TypeError):
            ActionPlan(steps=(Step(("true",)),),
                       reversibility="irreversible")

    def test_list_steps_coerced(self):
        plan = ActionPlan(steps=[Step(("true",))])
        self.assertIsInstance(plan.steps, tuple)


class TestRunner(unittest.TestCase):
    def test_read_only_plan_runs(self):
        res = run(ActionPlan(steps=(Step(("echo", "hi")),),
                             reversibility="read_only",
                             description="echo probe"))
        self.assertTrue(res["ok"])
        self.assertIn("hi", res["steps"][0]["stdout"])

    def test_failure_stops_the_plan(self):
        res = run(ActionPlan(steps=(Step(("true",)),
                                     Step(("false",)),
                                     Step(("echo", "never"))),
                             reversibility="read_only"))
        self.assertFalse(res["ok"])
        self.assertEqual(len(res["steps"]), 2)
        self.assertEqual(res["stopped_at"], ["false"])

    def test_timeout_is_an_honest_result(self):
        res = run_step(Step(("sleep", "5"), timeout_s=0.2))
        self.assertFalse(res["ok"])
        self.assertIn("timeout", res["error"])

    def test_postcondition_failure_auto_reverts(self):
        calls = []
        real_run = runner_mod.run_step

        def spy(step):
            calls.append(step.argv[0])
            return real_run(step)

        revert_plan = ActionPlan(
            steps=(Step(("true",)),), reversibility="read_only",
            description="revert")
        plan = ActionPlan(
            steps=(Step(("true",)),),
            reversibility="reversible", blast_radius="file",
            description="mutate then fail postcondition",
            postcondition_argv=("false",),
            revert=revert_plan)
        with mock.patch.object(runner_mod, "run_step", spy):
            res = run(plan)
        self.assertFalse(res["ok"])
        self.assertTrue(res["reverted"])
        self.assertIn("revert", res)

    def test_missing_binary_is_honest(self):
        res = run_step(Step(("/nonexistent-binary-xyz",)))
        self.assertFalse(res["ok"])
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()
