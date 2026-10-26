"""python -m step3_reward_verifier --input ... --response ... / --api"""
import argparse
import json
from pathlib import Path

from . import APIRewardModelVerifier, ReplayScorer, VerificationCase, VerifierModelUnavailable, verify_case


def main() -> None:
    parser = argparse.ArgumentParser(description="独立运行 Step 3 奖励模型终验")
    parser.add_argument("--input", required=True, type=Path, help="VerificationCase JSON 输入")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--response", type=Path, help="回放已有评分 JSON，不调用模型")
    mode.add_argument("--api", action="store_true", help="明确启用真实评分 API")
    parser.add_argument("--model", help="评分模型 ID")
    parser.add_argument("--env-file", type=Path, help="API 模式显式加载的配置文件")
    parser.add_argument("--output", type=Path, help="保存实验输出 JSON；不指定则打印")
    args = parser.parse_args()
    try:
        case = VerificationCase.model_validate_json(args.input.read_text(encoding="utf-8-sig"))
        if args.response:
            scorer = ReplayScorer(json.loads(args.response.read_text(encoding="utf-8-sig")))
        else:
            if args.env_file:
                if not args.env_file.is_file():
                    raise ValueError("指定的环境配置文件不存在")
                from dotenv import load_dotenv
                load_dotenv(args.env_file, override=False)
            if not case.source_knowledge:
                raise ValueError("API 评分输入需要 source_knowledge，必须由上游提供该候选的标准知识快照")
            scorer = APIRewardModelVerifier(
                model=args.model,
                knowledge_provider=lambda disorder: case.source_knowledge if disorder == case.candidate.disorder else {},
            )
        result = verify_case(case, scorer)
        result["scoring_mode"] = "replay" if args.response else "api"
        serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
            print(f"Saved: {args.output}")
        else:
            print(serialized, end="")
    except (OSError, ValueError, VerifierModelUnavailable) as exc:
        parser.exit(2, f"Step 3 运行失败：{exc}\n")


if __name__ == "__main__":
    main()
