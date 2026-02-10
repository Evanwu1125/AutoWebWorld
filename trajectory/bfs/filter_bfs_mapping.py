#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BFS Mapping 筛选工具

功能：
1. 收集旧 bfs_mapping 的所有文件路径（作为白名单）
2. 筛选新生成的 bfs_mapping，只保留白名单中的文件
3. 生成详细报告，显示保留、删除、丢失的文件

使用场景：
当 fsm.json 的 selector 被修改后，需要重新生成 BFS 以获取最新的 selector，
但只想保留原有的轨迹范围，不增加新的轨迹。
"""

import argparse
import sys
from pathlib import Path
from typing import Set, Tuple, List, Dict


def normalize_path(path: str) -> str:
    """
    规范化路径，只保留最后两级（目标状态/文件名）

    这样可以忽略顶层目录的差异（如 home_initial），只关注实际的目标状态和文件名。

    例如:
    - home_initial/SESSION_COMPLETED_SUCCESS/macro_001.json
      → SESSION_COMPLETED_SUCCESS/macro_001.json
    - SESSION_COMPLETED_SUCCESS/macro_001.json
      → SESSION_COMPLETED_SUCCESS/macro_001.json

    Args:
        path: 原始相对路径

    Returns:
        规范化后的路径（最后两级）
    """
    parts = Path(path).parts
    if len(parts) >= 2:
        return str(Path(parts[-2]) / parts[-1])
    return path


def collect_file_paths(bfs_mapping_dir: Path) -> Set[str]:
    """
    收集 bfs_mapping 目录下所有 JSON 文件的规范化路径

    Args:
        bfs_mapping_dir: bfs_mapping 目录路径

    Returns:
        规范化文件路径的集合（只保留最后两级：目标状态/文件名）
        例如: {"SESSION_COMPLETED_SUCCESS/macro_002_00_HOME__SESSION_COMPLETED_SUCCESS.json"}
    """
    if not bfs_mapping_dir.exists():
        print(f"[ERROR] 目录不存在: {bfs_mapping_dir}")
        return set()

    files = set()
    for json_file in bfs_mapping_dir.rglob("*.json"):
        try:
            rel_path = json_file.relative_to(bfs_mapping_dir)
            normalized = normalize_path(str(rel_path))
            files.add(normalized)
        except ValueError:
            continue

    return files


def filter_bfs_mapping(
    old_mapping: Path,
    new_mapping: Path,
    verbose: bool = True
) -> Tuple[int, int, List[str]]:
    if verbose:
        print(f"[INFO] 收集旧 bfs_mapping 的文件列表...")
        print(f"       旧目录: {old_mapping}")

    # 收集旧文件列表（白名单）- 使用规范化路径
    old_files = collect_file_paths(old_mapping)

    if not old_files:
        print(f"[ERROR] 旧 bfs_mapping 中没有找到任何文件")
        return 0, 0, []

    if verbose:
        print(f"[INFO] 旧 bfs_mapping 共有 {len(old_files)} 个文件")
        print(f"[INFO] 筛选新 bfs_mapping...")
        print(f"       新目录: {new_mapping}")

    kept = 0
    removed = 0
    new_files_normalized = set()  # 收集新文件的规范化路径

    # 遍历新生成的文件，删除不在白名单中的
    for json_file in new_mapping.rglob("*.json"):
        try:
            rel_path = str(json_file.relative_to(new_mapping))
            normalized = normalize_path(rel_path)
            new_files_normalized.add(normalized)

            if normalized in old_files:
                # 在白名单中，保留
                kept += 1
            else:
                # 不在白名单中，删除
                json_file.unlink()
                removed += 1
                if verbose and removed <= 5:
                    print(f"       删除: {rel_path}")
        except Exception as e:
            print(f"[WARNING] 处理文件失败: {json_file} -> {e}")

    # 检查哪些旧文件在新 mapping 中找不到（丢失）
    missing = []
    for old_file in old_files:
        if old_file not in new_files_normalized:
            missing.append(old_file)
    
    # 打印统计信息
    print("")
    print("=" * 60)
    print("筛选结果统计")
    print("=" * 60)
    print(f"✅ 保留文件: {kept}")
    print(f"❌ 删除文件: {removed}")
    print(f"⚠️  丢失文件: {len(missing)}")
    
    # 计算丢失率
    if old_files:
        loss_rate = len(missing) / len(old_files) * 100
        print(f"📊 丢失率: {loss_rate:.1f}%")
        
        if loss_rate > 50:
            print("")
            print("⚠️  警告：丢失率超过 50%！")
    
    # 显示丢失的文件（前 20 个）
    if missing:
        print("")
        print(f"丢失的文件列表（共 {len(missing)} 个）：")
        for i, f in enumerate(missing[:20], 1):
            print(f"  {i}. {f}")
        
        if len(missing) > 20:
            print(f"  ... 还有 {len(missing) - 20} 个文件未显示")
    
    print("=" * 60)
    
    return kept, removed, missing


def main():
    parser = argparse.ArgumentParser(
        description="筛选 BFS mapping，只保留在参考 mapping 中存在的文件"
    )
    parser.add_argument(
        "--old-mapping",
        type=Path,
        required=True,
        help="旧的 bfs_mapping 目录（作为白名单）"
    )
    parser.add_argument(
        "--new-mapping",
        type=Path,
        required=True,
        help="新生成的 bfs_mapping 目录（将被筛选）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只显示统计信息"
    )
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not args.old_mapping.exists():
        print(f"[ERROR] 旧 bfs_mapping 目录不存在: {args.old_mapping}")
        sys.exit(1)
    
    if not args.new_mapping.exists():
        print(f"[ERROR] 新 bfs_mapping 目录不存在: {args.new_mapping}")
        sys.exit(1)
    
    # 执行筛选
    kept, removed, missing = filter_bfs_mapping(
        args.old_mapping,
        args.new_mapping,
        verbose=not args.quiet
    )
    
    # 返回状态码
    if len(missing) > 0:
        sys.exit(2)  # 有丢失文件，但不算错误
    else:
        sys.exit(0)  # 成功


if __name__ == "__main__":
    main()

