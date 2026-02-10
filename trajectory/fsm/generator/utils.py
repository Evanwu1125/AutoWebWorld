import os
import re
import json
import time
from typing import Dict, Any


def slugify_theme(theme: str) -> str:
    """将主题名转换为文件系统安全的slug"""
    return re.sub(r'[^\w\-]', '_', theme.lower())


def save_json(data: Dict[str, Any], filepath: str):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_cost_report(themed_output_dir: str, 
                    theme: str,
                    generator_summary: Dict[str, Any],
                    validator_summary: Dict[str, Any],
                    improver_summary: Dict[str, Any]):
    """保存成本报告"""
    print("\n" + "=" * 80)
    print("💰 总成本统计")
    print("=" * 80)

    total_cost_usd = (generator_summary['total_cost_usd'] +
                      validator_summary['total_cost_usd'] +
                      improver_summary['total_cost_usd'])
    total_cost_cny = total_cost_usd * 7.3
    total_tokens = (generator_summary['total_tokens'] +
                    validator_summary['total_tokens'] +
                    improver_summary['total_tokens'])

    print(f"Generator Agent: ${generator_summary['total_cost_usd']:.4f} ({generator_summary['total_calls']} 次调用)")
    print(f"Validator Agent: ${validator_summary['total_cost_usd']:.4f} ({validator_summary['total_calls']} 次调用)")
    print(f"Improver Agent: ${improver_summary['total_cost_usd']:.4f} ({improver_summary['total_calls']} 次调用)")
    print(f"\n总成本: ${total_cost_usd:.4f} (¥{total_cost_cny:.2f})")
    print(f"总 tokens: {total_tokens:,}")
    print("=" * 80 + "\n")

    cost_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theme": theme,
        "generator": generator_summary,
        "validator": validator_summary,
        "improver": improver_summary,
        "total": {
            "cost_usd": total_cost_usd,
            "cost_cny": total_cost_cny,
            "tokens": total_tokens
        }
    }

    cost_report_path = f"{themed_output_dir}/cost_report.json"
    save_json(cost_report, cost_report_path)
    print(f"💾 成本报告已保存: {cost_report_path}\n")

