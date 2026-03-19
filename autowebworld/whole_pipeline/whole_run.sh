#!/bin/bash
# whole_run.sh

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "Error: OPENAI_API_KEY is not set"
  exit 1
fi


START_STAGE=0
MAX_WORKERS=40
HEADLESS=true

CAPTION_MODEL="deepseek-v3.2-exp"

IMAGE_MODEL="gemini-2.5-flash-image"

EXPANSION_MODEL="deepseek-v3.2-exp"

VISUAL_QUERY_MODEL="deepseek-v3.2-exp"

QA_LLM_MODEL="deepseek-v3.2-exp"
QA_VLM_MODEL="gemini-2.5-flash"

EXPANSION_CONCURRENCY=20
POSTPROCESS_CONCURRENCY=20

MAX_QUESTIONS=9
QA_CONCURRENCY=10
SKIP_VLM=false

PROJECT_LISTS=(
    # "travel_real_estate_projects.txt"
    # "media_learning_wellness_projects.txt"
    # "commerce_finance_projects.txt"
    # "communication_social_projects.txt"
    "productivity_projects.txt"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
PIPELINE_DATA_AUG_DIR="$PROJECT_ROOT/env_generator/data_augmentation"
FINISHED_LIST_DIR="$PROJECT_ROOT/finished_lists"

log() {
    local level="$1"
    shift
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    case "$level" in
        INFO)    echo "[$timestamp] ℹ️  $*" ;;
        SUCCESS) echo "[$timestamp] ✅ $*" ;;
        ERROR)   echo "[$timestamp] ❌ $*" ;;
        WARNING) echo "[$timestamp] ⚠️  $*" ;;
        STEP)    echo "[$timestamp] 🚀 $*" ;;
    esac
}

process_single_project() {
    local PROJECT_DIR="$1"
    local CATEGORY_NAME="$2"
    local PROJECT_NAME=$(basename "$(dirname "$PROJECT_DIR")")

    log STEP "Start processing project: $PROJECT_NAME"
    log INFO "Project path: $PROJECT_DIR"
    
    if [ ! -d "$PROJECT_DIR" ]; then
        log ERROR "Project directory does not exist: $PROJECT_DIR"
        return 1
    fi
    
    if [ ! -f "$PROJECT_DIR/fsm.json" ]; then
        log ERROR "Missing fsm.json: $PROJECT_DIR"
        return 1
    fi
    
    if [ ! -f "$PROJECT_DIR/src/stores/data.js" ]; then
        log ERROR "Missing data.js: $PROJECT_DIR"
        return 1
    fi
    
    local EXPANSION_DIR="$PROJECT_ROOT/env_generator/data_augmentation/item_expansion"
    local GROUNDING_DIR="$PROJECT_ROOT/web_extractor/grounding_extract"
    local OUTPUT_BASE="$PROJECT_ROOT/new_outputs/$CATEGORY_NAME/$PROJECT_NAME"
    local VISUAL_QUERY_OUTPUT="$OUTPUT_BASE/0_visual_query"
    local EXPANDED_OUTPUT="$OUTPUT_BASE/1_expanded"
    local GROUNDING_OUTPUT="$OUTPUT_BASE/2_grounding"
    local QA_QUERY_OUTPUT="$OUTPUT_BASE/3_qa_query"
    local INTERMEDIATE_DIR="$OUTPUT_BASE/intermediate"
    local LOGS_DIR="$OUTPUT_BASE/logs"
    local ERROR_LOG="$LOGS_DIR/error_log.txt"

    mkdir -p "$OUTPUT_BASE"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$INTERMEDIATE_DIR"

    if [ "$START_STAGE" -le 0 ]; then
        log STEP "[$PROJECT_NAME] Stage 0/6: Auto-replace DateTimePicker component"

        local TARGET_DATETIMEPICKER="$PROJECT_DIR/src/components/widgets/DateTimePicker.vue"
        local FSM_FILE="$PROJECT_DIR/fsm.json"

        if [ ! -f "$TARGET_DATETIMEPICKER" ]; then
            log WARNING "[$PROJECT_NAME] Target DateTimePicker.vue not found, skipping replacement"
        else
            local dp_version=$(python "$PROJECT_ROOT/detect_datepicker_version.py" "$FSM_FILE")
            
            case "$dp_version" in
                "full")
                    local SOURCE_DATETIMEPICKER="$PROJECT_ROOT/custom_components/DateTimePicker.vue"
                    if [ -f "$SOURCE_DATETIMEPICKER" ]; then
                        cp "$SOURCE_DATETIMEPICKER" "$TARGET_DATETIMEPICKER" || {
                            log ERROR "[$PROJECT_NAME] Failed to copy DateTimePicker.vue"
                            return 1
                        }
                        log SUCCESS "[$PROJECT_NAME] Replaced DateTimePicker.vue (full version: date + time)"
                    else
                        log ERROR "[$PROJECT_NAME] Full DateTimePicker not found: $SOURCE_DATETIMEPICKER"
                        return 1
                    fi
                    ;;
                "date_only")
                    local SOURCE_DATETIMEPICKER="$PROJECT_ROOT/custom_components/DateTimePickerDateOnly.vue"
                    if [ -f "$SOURCE_DATETIMEPICKER" ]; then
                        cp "$SOURCE_DATETIMEPICKER" "$TARGET_DATETIMEPICKER" || {
                            log ERROR "[$PROJECT_NAME] Failed to copy DateTimePickerDateOnly.vue"
                            return 1
                        }
                        log SUCCESS "[$PROJECT_NAME] Replaced DateTimePicker.vue (date-only version)"
                    else
                        log ERROR "[$PROJECT_NAME] Date-only DateTimePicker not found: $SOURCE_DATETIMEPICKER"
                        return 1
                    fi
                    ;;
                *)
                    log WARNING "[$PROJECT_NAME] Failed to detect DateTimePicker version: $dp_version, using default full version"
                    local SOURCE_DATETIMEPICKER="$PROJECT_ROOT/custom_components/DateTimePicker.vue"
                    if [ -f "$SOURCE_DATETIMEPICKER" ]; then
                        cp "$SOURCE_DATETIMEPICKER" "$TARGET_DATETIMEPICKER" || {
                            log ERROR "[$PROJECT_NAME] Failed to copy DateTimePicker.vue"
                            return 1
                        }
                        log SUCCESS "[$PROJECT_NAME] Replaced DateTimePicker.vue (full version: date + time) [default]"
                    else
                        log ERROR "[$PROJECT_NAME] Full DateTimePicker not found: $SOURCE_DATETIMEPICKER"
                        return 1
                    fi
                    ;;
            esac
        fi
    fi

    if [ "$START_STAGE" -le 1 ]; then
        local VISUAL_OUTPUT_WEB="$VISUAL_QUERY_OUTPUT/web"
        if [ -d "$VISUAL_OUTPUT_WEB" ] && [ -f "$VISUAL_OUTPUT_WEB/caption.json" ]; then
            log INFO "[$PROJECT_NAME] Stage 1 output already exists, skipping image generation"
            log INFO "[$PROJECT_NAME] Output directory: $VISUAL_QUERY_OUTPUT"
        else
            log STEP "[$PROJECT_NAME] Stage 1/6: Visual Query - image generation and deployment"
            
            echo "" >> "$ERROR_LOG"
            echo "========================================" >> "$ERROR_LOG"
            echo "Project: $PROJECT_NAME" >> "$ERROR_LOG"
            echo "Stage 1: Visual Query - Images" >> "$ERROR_LOG"
            echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG"
            echo "========================================" >> "$ERROR_LOG"
            
            cd "$PROJECT_ROOT"
            python -m env_generator.data_augmentation.visual_query.main \
                --data-path "$PROJECT_DIR" \
                --bfs-dir "" \
                --output-dir "$VISUAL_QUERY_OUTPUT" \
                --stage 1 \
                --skip-stages 3 \
                --caption-model "$CAPTION_MODEL" \
                --image-model "$IMAGE_MODEL" || {
                    log ERROR "[$PROJECT_NAME] Visual Query image generation failed"
                    return 1
                }

            log SUCCESS "[$PROJECT_NAME] Visual Query image generation and deployment completed"
            log INFO "[$PROJECT_NAME] Output directory: $VISUAL_QUERY_OUTPUT"
        fi
    fi

    if [ "$START_STAGE" -le 2 ]; then
        log STEP "[$PROJECT_NAME] Stage 2/6: Item Expansion"

        local MOCKDATA_JS="$PROJECT_DIR/src/stores/data.js"
        local FSM_FILE="$PROJECT_DIR/fsm.json"
        local MOCKDATA_JSON="$INTERMEDIATE_DIR/mockdata.json"
        local FSM_SPLIT="$INTERMEDIATE_DIR/fsm_split.json"
        local FSM_NORM="$INTERMEDIATE_DIR/fsm_norm.json"
        local BFS_ALLSHORTEST="$INTERMEDIATE_DIR/allshortest.json"
        local BFS_MAPPING="$INTERMEDIATE_DIR/bfs_mapping"

        echo "" >> "$ERROR_LOG"
        echo "========================================" >> "$ERROR_LOG"
        echo "Stage 2: Item Expansion" >> "$ERROR_LOG"
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG"
        echo "========================================" >> "$ERROR_LOG"

        log INFO "[$PROJECT_NAME] Regenerating BFS mapping from the latest fsm.json..."

        cd "$PROJECT_ROOT/env_generator/bfs_generator"
        log INFO "[$PROJECT_NAME] 2.1 Split filters..."
        python split_filters.py --input "$FSM_FILE" --output "$FSM_SPLIT" || return 1

        log INFO "[$PROJECT_NAME] 2.2 Normalize FSM..."
        python normalize.py --input "$FSM_SPLIT" --output "$FSM_NORM" || return 1

        log INFO "[$PROJECT_NAME] 2.3 BFS search..."
        python bfs_action.py --fsm "$FSM_SPLIT" --norm "$FSM_NORM" --out "$BFS_ALLSHORTEST" || return 1

        log INFO "[$PROJECT_NAME] 2.4 GUI mapping..."
        python gui_mapping.py --fsm "$FSM_SPLIT" --bfs "$BFS_ALLSHORTEST" --out "$BFS_MAPPING" || return 1

        log SUCCESS "[$PROJECT_NAME] BFS generation completed"
        log INFO "[$PROJECT_NAME] Using the full BFS generated from the current fsm.json"

        # Preprocess slider
        cd "$EXPANSION_DIR"
        log INFO "[$PROJECT_NAME] 2.6 Preprocess sliders..."
        python preprocess_slider.py --input "$BFS_MAPPING" || return 1

        # Convert mockdata
        cd "$EXPANSION_DIR/../scripts"
        log INFO "[$PROJECT_NAME] 2.7 Convert mockdata..."
        python convert_data_js.py "$MOCKDATA_JS" "$MOCKDATA_JSON" || return 1

        # Item expansion
        rm -rf "$EXPANDED_OUTPUT"
        mkdir -p "$EXPANDED_OUTPUT"

        cd "$PROJECT_ROOT"
        log INFO "[$PROJECT_NAME] 2.8 Item expansion..."
        python -m env_generator.data_augmentation.item_expansion.main \
            --input "$BFS_MAPPING" \
            --mockdata "$MOCKDATA_JSON" \
            --output "$EXPANDED_OUTPUT" \
            --concurrency "$EXPANSION_CONCURRENCY" \
            --error-log "$ERROR_LOG" || return 1

        log INFO "[$PROJECT_NAME] 2.9 Post-processing..."
        python -m env_generator.data_augmentation.item_expansion.post_process \
            --input "$EXPANDED_OUTPUT" \
            --mockdata "$MOCKDATA_JSON" \
            --model "$EXPANSION_MODEL" \
            --concurrency "$POSTPROCESS_CONCURRENCY" \
            --error-log "$ERROR_LOG" || return 1

        log SUCCESS "[$PROJECT_NAME] Item Expansion completed"
        log INFO "[$PROJECT_NAME] Output directory: $EXPANDED_OUTPUT"
    fi

    # ========== Stage 3: Grounding Extract ==========
    if [ "$START_STAGE" -le 3 ]; then
        log STEP "[$PROJECT_NAME] Stage 3/6: Grounding Extract"

        if [ -d "$GROUNDING_OUTPUT" ]; then
            log INFO "[$PROJECT_NAME] Cleaning previous grounding output..."
            rm -rf "$GROUNDING_OUTPUT"
        fi
        mkdir -p "$GROUNDING_OUTPUT"

        echo "" >> "$ERROR_LOG"
        echo "========================================" >> "$ERROR_LOG"
        echo "Stage 3: Grounding Extract" >> "$ERROR_LOG"
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG"
        echo "========================================" >> "$ERROR_LOG"

        log INFO "[$PROJECT_NAME] Cleaning and reinstalling frontend dependencies..."

        if [ -d "$PROJECT_DIR/node_modules" ]; then
            log INFO "[$PROJECT_NAME] Removing existing node_modules..."
            rm -rf "$PROJECT_DIR/node_modules"
        fi

        if ! command -v pnpm &> /dev/null; then
            log ERROR "[$PROJECT_NAME] pnpm is not installed. Please install pnpm first"
            return 1
        fi

        log INFO "[$PROJECT_NAME] Running pnpm install..."
        cd "$PROJECT_DIR"
        if [ -f "pnpm-lock.yaml" ]; then
            pnpm install --frozen-lockfile || {
                log ERROR "[$PROJECT_NAME] pnpm install failed"
                return 1
            }
        else
            pnpm install || {
                log ERROR "[$PROJECT_NAME] pnpm install failed"
                return 1
            }
        fi

        log SUCCESS "[$PROJECT_NAME] Dependency installation completed"

        if [ ! -d "$EXPANDED_OUTPUT" ]; then
            log ERROR "[$PROJECT_NAME] Expanded output directory does not exist: $EXPANDED_OUTPUT"
            return 1
        fi

        cd "$GROUNDING_DIR"

        local CMD="python -m scripts.run_batch \
            \"$EXPANDED_OUTPUT\" \
            \"$PROJECT_DIR\" \
            --output-dir \"$GROUNDING_OUTPUT\" \
            --max-workers $MAX_WORKERS \
            --scroll-step 50 \
            --num-slider-steps 3 \
            --viewport-width 1280 \
            --viewport-height 720 \
            --max-scroll-iterations 50"

        [ "$HEADLESS" = "true" ] && CMD="$CMD --headless"

        log INFO "[$PROJECT_NAME] Running grounding extract..."
        eval $CMD || return 1

        log SUCCESS "[$PROJECT_NAME] Grounding Extract completed"
        log INFO "[$PROJECT_NAME] Output directory: $GROUNDING_OUTPUT"

        echo "" >> "$ERROR_LOG"
        echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG"
        echo "Status: ✅ Success" >> "$ERROR_LOG"
    fi

    if [ "$START_STAGE" -le 4 ]; then
        log STEP "[$PROJECT_NAME] Stage 4/6: Visual Query - generate visual queries"

        echo "" >> "$ERROR_LOG"
        echo "========================================" >> "$ERROR_LOG"
        echo "Stage 4: Visual Query Generation" >> "$ERROR_LOG"
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG"
        echo "========================================" >> "$ERROR_LOG"

        cd "$PROJECT_ROOT"
        python -m env_generator.data_augmentation.visual_query.main \
            --data-path "$PROJECT_DIR" \
            --bfs-dir "$GROUNDING_OUTPUT" \
            --output-dir "$VISUAL_QUERY_OUTPUT" \
            --stage 3 \
            --visual-query-model "$VISUAL_QUERY_MODEL" || {
                log ERROR "[$PROJECT_NAME] Visual Query generation failed"
                return 1
            }

        log SUCCESS "[$PROJECT_NAME] Visual Query generation completed"
        log INFO "[$PROJECT_NAME] Output directory: $VISUAL_QUERY_OUTPUT/visual_queries"
    fi

    if [ "$START_STAGE" -le 5 ]; then
        log STEP "[$PROJECT_NAME] Stage 5/6: QA Query generation"

        echo "" >> "$ERROR_LOG"
        echo "========================================" >> "$ERROR_LOG"
        echo "Stage 5: QA Query Generation" >> "$ERROR_LOG"
        echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG"
        echo "========================================" >> "$ERROR_LOG"

        local CAPTION_FILE="$VISUAL_QUERY_OUTPUT/web/caption.json"
        if [ ! -f "$CAPTION_FILE" ]; then
            log ERROR "[$PROJECT_NAME] Caption file does not exist: $CAPTION_FILE"
            return 1
        fi

        if [ ! -d "$GROUNDING_OUTPUT" ]; then
            log ERROR "[$PROJECT_NAME] Grounding output directory does not exist: $GROUNDING_OUTPUT"
            return 1
        fi

        mkdir -p "$QA_QUERY_OUTPUT"

        cd "$PROJECT_ROOT"
        local QA_CMD="python -m env_generator.data_augmentation.qa_query.main \
            --grounding-dir \"$GROUNDING_OUTPUT\" \
            --caption-file \"$CAPTION_FILE\" \
            --output-dir \"$QA_QUERY_OUTPUT\" \
            --llm-model \"$QA_LLM_MODEL\" \
            --vlm-model \"$QA_VLM_MODEL\" \
            --max-questions $MAX_QUESTIONS \
            --concurrency $QA_CONCURRENCY"

        [ "$SKIP_VLM" = "true" ] && QA_CMD="$QA_CMD --skip-vlm"

        log INFO "[$PROJECT_NAME] Running QA Query generation..."
        eval $QA_CMD || {
            log ERROR "[$PROJECT_NAME] QA Query generation failed"
            return 1
        }

        log SUCCESS "[$PROJECT_NAME] QA Query generation completed"
        log INFO "[$PROJECT_NAME] Output directory: $QA_QUERY_OUTPUT"

        echo "" >> "$ERROR_LOG"
        echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$ERROR_LOG"
        echo "Status: ✅ Success" >> "$ERROR_LOG"
    fi

    log SUCCESS "[$PROJECT_NAME] All stages completed!"
    log INFO "[$PROJECT_NAME] Final output directory: $OUTPUT_BASE"

    log INFO "[$PROJECT_NAME] Collecting trajectory statistics..."
    python3 - <<STATS_EOF
import json
from pathlib import Path
from datetime import datetime

output_base = Path("$OUTPUT_BASE")
stats = {
    "project_name": "$PROJECT_NAME",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "stages": {}
}

expanded_dir = output_base / "1_expanded"
if expanded_dir.exists():
    expanded_count = len(list(expanded_dir.rglob("*.json")))
    stats["stages"]["stage2_item_expansion"] = {
        "trajectory_count": expanded_count,
        "description": "Expanded trajectories (item-specific trajectories)"
    }

grounding_dir = output_base / "2_grounding"
if grounding_dir.exists():
    grounding_count = len([d for d in grounding_dir.iterdir() if d.is_dir()])
    stats["stages"]["stage3_grounding"] = {
        "trajectory_count": grounding_count,
        "description": "Grounding extraction results"
    }

visual_query_dir = output_base / "0_visual_query" / "web" / "visual_queries"
if visual_query_dir.exists():
    visual_query_count = len(list(visual_query_dir.rglob("*.json")))
    stats["stages"]["stage4_visual_query"] = {
        "query_count": visual_query_count,
        "description": "Visual Query generation results"
    }

qa_query_dir = output_base / "3_qa_query"
summary_file = qa_query_dir / "summary.json"
if summary_file.exists():
    with open(summary_file) as f:
        summary = json.load(f)

    items_processed = summary.get("items_processed", 0)
    total_qa_pairs = summary.get("total_qa_pairs", 0)

    stats["stages"]["stage5_qa_query"] = {
        "trajectory_count": items_processed,
        "qa_pairs_count": total_qa_pairs,
        "description": f"QA Query generation results ({items_processed} trajectories, {total_qa_pairs} QA pairs)"
    }
elif qa_query_dir.exists():
    qa_trajectory_count = len([d for d in qa_query_dir.iterdir() if d.is_dir() and d.name.startswith('macro_')])
    stats["stages"]["stage5_qa_query"] = {
        "trajectory_count": qa_trajectory_count,
        "description": "QA Query generation results"
    }

stats_file = output_base / "trajectory_stats.json"
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"Trajectory statistics:")
for stage, data in stats["stages"].items():
    if "qa_pairs_count" in data:
        print(f"  {stage}: {data['trajectory_count']} trajectories, {data['qa_pairs_count']} QA pairs")
    elif "trajectory_count" in data:
        print(f"  {stage}: {data['trajectory_count']} trajectories")
    elif "query_count" in data:
        print(f"  {stage}: {data['query_count']} queries")
print(f"Statistics saved to: {stats_file}")
print(f"{'='*60}")
STATS_EOF

    python3 - <<EOF
import json
from pathlib import Path

output_base = Path("$OUTPUT_BASE")
cost_files = {
    'stage1_caption': output_base / '0_visual_query' / 'web' / 'api_cost_stage1_caption.json',
    'stage1_image': output_base / '0_visual_query' / 'web' / 'api_cost_stage1_image.json',
    'stage2_expansion': output_base / 'api_cost_stage2_expansion.json',
    'stage2_postprocess': output_base / 'api_cost_stage2_postprocess.json',
    'stage4_visual_query': output_base / '0_visual_query' / 'web' / 'api_cost_stage4_visual_query.json',
    'stage5_qa_llm': output_base / '3_qa_query' / 'api_cost_llm.json',
    'stage5_qa_vlm': output_base / '3_qa_query' / 'api_cost_vlm.json'
}

total_stats = {
    'input_tokens': 0,
    'output_tokens': 0,
    'total_tokens': 0,
    'input_cost_usd': 0.0,
    'output_cost_usd': 0.0,
    'total_cost_usd': 0.0,
    'stages': {}
}

for stage_name, cost_file in cost_files.items():
    if cost_file.exists():
        with open(cost_file) as f:
            stats = json.load(f)

            if stage_name == 'stage1_image':
                input_tokens = stats.get('total_input_tokens', 0)
                output_tokens = stats.get('total_output_tokens', 0)
                input_cost = stats.get('input_cost_usd', 0.0)
                output_cost = stats.get('output_cost_usd', 0.0)
                total_cost = stats.get('total_cost', 0.0)

                total_stats['input_tokens'] += input_tokens
                total_stats['output_tokens'] += output_tokens
                total_stats['total_tokens'] += stats.get('total_tokens', 0)
                total_stats['input_cost_usd'] += input_cost
                total_stats['output_cost_usd'] += output_cost
                total_stats['total_cost_usd'] += total_cost
            else:
                total_stats['input_tokens'] += stats.get('input_tokens', 0)
                total_stats['output_tokens'] += stats.get('output_tokens', 0)
                total_stats['total_tokens'] += stats.get('total_tokens', 0)
                total_stats['input_cost_usd'] += stats.get('input_cost_usd', 0.0)
                total_stats['output_cost_usd'] += stats.get('output_cost_usd', 0.0)
                total_stats['total_cost_usd'] += stats.get('total_cost_usd', 0.0)

            total_stats['stages'][stage_name] = stats
    else:
        print(f"Warning: Cost file not found: {cost_file}")

total_navigation_cost = 0.0
total_visual_cost = 0.0
total_qa_cost = 0.0

# Navigation cost: stage2_expansion + stage2_postprocess
if 'stage2_expansion' in total_stats['stages']:
    total_navigation_cost += total_stats['stages']['stage2_expansion'].get('total_cost_usd', 0.0)
if 'stage2_postprocess' in total_stats['stages']:
    total_navigation_cost += total_stats['stages']['stage2_postprocess'].get('total_cost_usd', 0.0)

# Visual cost: stage1_caption + stage1_image + stage4_visual_query
if 'stage1_caption' in total_stats['stages']:
    total_visual_cost += total_stats['stages']['stage1_caption'].get('total_cost_usd', 0.0)
if 'stage1_image' in total_stats['stages']:
    total_visual_cost += total_stats['stages']['stage1_image'].get('total_cost', 0.0)
if 'stage4_visual_query' in total_stats['stages']:
    total_visual_cost += total_stats['stages']['stage4_visual_query'].get('total_cost_usd', 0.0)

# QA cost: stage5_qa_llm + stage5_qa_vlm
if 'stage5_qa_llm' in total_stats['stages']:
    total_qa_cost += total_stats['stages']['stage5_qa_llm'].get('total_cost_usd', 0.0)
if 'stage5_qa_vlm' in total_stats['stages']:
    total_qa_cost += total_stats['stages']['stage5_qa_vlm'].get('total_cost_usd', 0.0)

total_stats['total_navigation_cost'] = round(total_navigation_cost, 6)
total_stats['total_visual_cost'] = round(total_visual_cost, 6)
total_stats['total_qa_cost'] = round(total_qa_cost, 6)

total_file = output_base / 'api_cost_total.json'
with open(total_file, 'w') as f:
    json.dump(total_stats, f, indent=2)

print(f"\n{'='*60}")
print(f"API Cost Summary:")
print(f"  Total input tokens:  {total_stats['input_tokens']:,}")
print(f"  Total output tokens: {total_stats['output_tokens']:,}")
print(f"  Total cost:          \${total_stats['total_cost_usd']:.6f}")
print(f"  Stages recorded:     {len(total_stats['stages'])}/7")
print(f"Cost summary saved to: {total_file}")
print(f"{'='*60}")
EOF

    return 0
}

batch_process() {
    local PROJECT_LIST="$1"
    mkdir -p "$FINISHED_LIST_DIR"
    local FINISHED_LIST="$FINISHED_LIST_DIR/$(basename "${PROJECT_LIST%.txt}")_finished.txt"

    local CATEGORY_NAME=$(basename "${PROJECT_LIST%.txt}")

    touch "$FINISHED_LIST"

    local total_projects=$(grep -v '^#' "$PROJECT_LIST" | grep -v '^$' | wc -l)
    local success_count=0
    local failed_count=0
    local skipped_count=0
    local failed_projects=()

    log INFO "Start batch processing: $PROJECT_LIST"
    log INFO "Category name: $CATEGORY_NAME"
    log INFO "Total projects: $total_projects"
    echo ""

    local current=0
    while IFS= read -r project_dir; do
        [[ -z "$project_dir" || "$project_dir" =~ ^# ]] && continue

        if grep -Fxq "$project_dir" "$FINISHED_LIST"; then
            local project_name=$(basename "$(dirname "$(dirname "$project_dir")")")
            log INFO "Skip completed project: $project_name"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        current=$((current + 1))
        echo ""
        echo "=========================================="
        log STEP "Progress: [$current/$total_projects]"
        echo "=========================================="

        if process_single_project "$project_dir" "$CATEGORY_NAME"; then
            success_count=$((success_count + 1))
            echo "$project_dir" >> "$FINISHED_LIST"
        else
            failed_count=$((failed_count + 1))
            failed_projects+=("$(basename "$(dirname "$(dirname "$project_dir")")")")
        fi
    done < "$PROJECT_LIST"

    echo ""
    echo "=========================================="
    log STEP "Batch processing completed"
    echo "=========================================="
    log INFO "Total projects: $total_projects"
    log SUCCESS "Success:        $success_count"
    log ERROR "Failed:         $failed_count"
    log INFO "Skipped:        $skipped_count"

    if [ ${#failed_projects[@]} -gt 0 ]; then
        echo ""
        log ERROR "Failed projects:"
        for project in "${failed_projects[@]}"; do
            echo "  - $project"
        done
    fi
}

main() {
    log STEP "Launching Visual Query Pipeline"
    log INFO "Configuration:"
    log INFO "  - START_STAGE: $START_STAGE"
    log INFO "  - MAX_WORKERS: $MAX_WORKERS"
    log INFO "  - HEADLESS: $HEADLESS"
    log INFO "Model configuration:"
    log INFO "  - CAPTION_MODEL: $CAPTION_MODEL (Stage 1 Caption)"
    log INFO "  - IMAGE_MODEL: $IMAGE_MODEL (Stage 1 image generation)"
    log INFO "  - EXPANSION_MODEL: $EXPANSION_MODEL (Stage 2)"
    log INFO "  - VISUAL_QUERY_MODEL: $VISUAL_QUERY_MODEL (Stage 4)"
    log INFO "  - QA_LLM_MODEL: $QA_LLM_MODEL (Stage 5 question generation)"
    log INFO "  - QA_VLM_MODEL: $QA_VLM_MODEL (Stage 5 VLM verification)"
    log INFO "Concurrency configuration:"
    log INFO "  - EXPANSION_CONCURRENCY: $EXPANSION_CONCURRENCY (Stage 2 Expansion)"
    log INFO "  - POSTPROCESS_CONCURRENCY: $POSTPROCESS_CONCURRENCY (Stage 2 query generation)"
    log INFO "  - QA_CONCURRENCY: $QA_CONCURRENCY (Stage 5 QA generation)"
    echo ""

    log INFO "BFS strategy: no CSV or legacy mapping; generate the full BFS directly from the current project fsm.json"
    echo ""

    for project_list in "${PROJECT_LISTS[@]}"; do
        local list_path="$PIPELINE_DATA_AUG_DIR/$project_list"

        if [ ! -f "$list_path" ]; then
            log WARNING "Project list file does not exist, skipping: $list_path"
            continue
        fi

        batch_process "$list_path"
        echo ""
    done

    log SUCCESS "All batch runs completed!"
}

main "$@"

