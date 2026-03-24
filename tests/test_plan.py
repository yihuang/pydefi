"""Tests for pydefi.plan, pydefi.planner, and pydefi.cli"""

import pytest

from pydefi.plan import ActionType, BridgeAction, ExecutionPlan, SwapAction
from pydefi.planner import build_plan
from pydefi.types import ChainId

# ---------------------------------------------------------------------------
# ExecutionPlan tests
# ---------------------------------------------------------------------------


class TestExecutionPlan:
    def _make_plan(self) -> ExecutionPlan:
        plan = ExecutionPlan(description="Test plan")
        plan.add(
            SwapAction(
                chain_id=ChainId.ETHEREUM,
                token_in="DAI",
                token_out="USDC",
                amount_in="1000",
            )
        )
        plan.add(
            BridgeAction(
                src_chain_id=ChainId.ETHEREUM,
                dst_chain_id=ChainId.BASE,
                token="USDC",
                amount_in="1000",
            )
        )
        return plan

    def test_add_and_length(self):
        plan = self._make_plan()
        assert len(plan.actions) == 2

    def test_action_types(self):
        plan = self._make_plan()
        assert plan.actions[0].action_type == ActionType.SWAP
        assert plan.actions[1].action_type == ActionType.BRIDGE

    def test_describe_contains_steps(self):
        plan = self._make_plan()
        description = plan.describe()
        assert "Step 1" in description
        assert "Step 2" in description
        assert "DAI" in description
        assert "USDC" in description

    def test_roundtrip_json(self):
        plan = self._make_plan()
        restored = ExecutionPlan.from_json(plan.to_json())
        assert restored.description == plan.description
        assert len(restored.actions) == len(plan.actions)
        assert isinstance(restored.actions[0], SwapAction)
        assert isinstance(restored.actions[1], BridgeAction)

    def test_to_dict_has_action_type_field(self):
        plan = self._make_plan()
        d = plan.to_dict()
        assert d["actions"][0]["action_type"] == "swap"
        assert d["actions"][1]["action_type"] == "bridge"

    def test_from_dict_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown action_type"):
            ExecutionPlan.from_dict({"actions": [{"action_type": "unknown"}]})


class TestSwapAction:
    def test_describe_without_estimate(self):
        action = SwapAction(chain_id=ChainId.ETHEREUM, token_in="ETH", token_out="USDC", amount_in="1.0")
        desc = action.describe()
        assert "ETH" in desc
        assert "USDC" in desc
        assert str(ChainId.ETHEREUM) in desc

    def test_describe_with_estimate(self):
        action = SwapAction(
            chain_id=ChainId.ETHEREUM,
            token_in="ETH",
            token_out="USDC",
            amount_in="1.0",
            estimated_amount_out="3500",
        )
        desc = action.describe()
        assert "3500" in desc


class TestBridgeAction:
    def test_describe(self):
        action = BridgeAction(
            src_chain_id=ChainId.ETHEREUM,
            dst_chain_id=ChainId.BASE,
            token="USDC",
            amount_in="500",
            estimated_time_seconds=120,
        )
        desc = action.describe()
        assert "USDC" in desc
        assert "120s" in desc


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------


class TestBuildPlan:
    def test_same_chain_different_tokens(self):
        plan = build_plan(ChainId.ETHEREUM, "DAI", ChainId.ETHEREUM, "WETH", "100")
        assert len(plan.actions) == 1
        assert isinstance(plan.actions[0], SwapAction)
        assert plan.actions[0].token_in == "DAI"
        assert plan.actions[0].token_out == "WETH"

    def test_same_chain_same_token_produces_no_steps(self):
        plan = build_plan(ChainId.ETHEREUM, "USDC", ChainId.ETHEREUM, "USDC", "100")
        assert len(plan.actions) == 0

    def test_cross_chain_src_token_is_bridgeable(self):
        # USDC is bridgeable → no source swap needed
        plan = build_plan(ChainId.ETHEREUM, "USDC", ChainId.BASE, "WETH", "500")
        action_types = [a.action_type for a in plan.actions]
        # Should start with bridge then swap
        assert action_types[0] == ActionType.BRIDGE
        assert action_types[-1] == ActionType.SWAP

    def test_cross_chain_dst_token_is_bridgeable(self):
        # WETH is bridgeable → no destination swap needed
        plan = build_plan(ChainId.ETHEREUM, "DAI", ChainId.BASE, "WETH", "1000")
        action_types = [a.action_type for a in plan.actions]
        # Should start with swap then bridge
        assert action_types[0] == ActionType.SWAP
        assert action_types[-1] == ActionType.BRIDGE

    def test_cross_chain_full_path(self):
        # Neither token is directly bridgeable
        plan = build_plan(ChainId.ETHEREUM, "MKR", ChainId.ARBITRUM, "GMX", "1")
        action_types = [a.action_type for a in plan.actions]
        assert action_types == [ActionType.SWAP, ActionType.BRIDGE, ActionType.SWAP]

    def test_plan_description_contains_chain_names(self):
        plan = build_plan(ChainId.ETHEREUM, "USDC", ChainId.BASE, "WETH", "100")
        assert "Ethereum" in plan.description
        assert "Base" in plan.description

    def test_bridge_action_chain_ids(self):
        plan = build_plan(ChainId.ETHEREUM, "USDC", ChainId.ARBITRUM, "WETH", "100")
        bridge = next(a for a in plan.actions if isinstance(a, BridgeAction))
        assert bridge.src_chain_id == ChainId.ETHEREUM
        assert bridge.dst_chain_id == ChainId.ARBITRUM


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_plan_executes(self):
        from click.testing import CliRunner

        from pydefi.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plan",
                "--src-chain",
                "ethereum",
                "--src-token",
                "USDC",
                "--dst-chain",
                "base",
                "--dst-token",
                "WETH",
                "--amount",
                "100",
            ],
        )
        assert result.exit_code == 0
        assert "Ethereum" in result.output
        assert "Base" in result.output
        assert "USDC" in result.output

    def test_swap_executes(self):
        from click.testing import CliRunner

        from pydefi.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "swap",
                "--chain",
                "1",
                "--token-in",
                "DAI",
                "--token-out",
                "USDC",
                "--amount",
                "50",
            ],
        )
        assert result.exit_code == 0
        assert "DAI" in result.output
        assert "USDC" in result.output

    def test_bridge_executes(self):
        from click.testing import CliRunner

        from pydefi.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "bridge",
                "--src-chain",
                "eth",
                "--dst-chain",
                "arbitrum",
                "--token",
                "USDC",
                "--amount",
                "200",
            ],
        )
        assert result.exit_code == 0
        assert "USDC" in result.output

    def test_unknown_chain_fails(self):
        from click.testing import CliRunner

        from pydefi.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plan",
                "--src-chain",
                "notachain",
                "--src-token",
                "USDC",
                "--dst-chain",
                "base",
                "--dst-token",
                "WETH",
                "--amount",
                "100",
            ],
        )
        assert result.exit_code != 0

    def test_numeric_chain_id(self):
        from pydefi.cli import _resolve_chain

        assert _resolve_chain("8453") == ChainId.BASE

    def test_chain_alias_eth(self):
        from pydefi.cli import _resolve_chain

        assert _resolve_chain("eth") == ChainId.ETHEREUM

    def test_cmd_plan_output(self, tmp_path):
        from click.testing import CliRunner

        from pydefi.cli import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plan",
                "--src-chain",
                "ethereum",
                "--src-token",
                "DAI",
                "--dst-chain",
                "base",
                "--dst-token",
                "WETH",
                "--amount",
                "1000",
            ],
        )
        assert result.exit_code == 0
        assert "Ethereum" in result.output
        assert "Base" in result.output
        assert "swap" in result.output  # action_type in JSON

    def test_cmd_execute_dry_run(self, tmp_path):
        from click.testing import CliRunner

        from pydefi.cli import cli
        from pydefi.planner import build_plan

        plan = build_plan(ChainId.ETHEREUM, "DAI", ChainId.BASE, "WETH", "100")
        plan_file = tmp_path / "plan.json"
        plan_file.write_text(plan.to_json())

        runner = CliRunner()
        result = runner.invoke(cli, ["execute", "--plan", str(plan_file), "--dry-run"])
        assert result.exit_code == 0
        assert "dry-run" in result.output

    def test_plan_output_saved_to_file(self, tmp_path):
        from click.testing import CliRunner

        from pydefi.cli import cli

        out_file = tmp_path / "plan.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "plan",
                "--src-chain",
                "ethereum",
                "--src-token",
                "DAI",
                "--dst-chain",
                "base",
                "--dst-token",
                "WETH",
                "--amount",
                "100",
                "--output",
                str(out_file),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        import json

        data = json.loads(out_file.read_text())
        assert data["description"]
        assert len(data["actions"]) > 0
