#!/usr/bin/env bash
set -euo pipefail

# 默认值和命令行中的相对路径均以仓库根目录为基准。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
dataset="../specialist_datasets/pick_cup_02/processed/single/trial_261117_002"
session="20261117_182618_IAQEB260436VUM"
layer="state_s8_r100"
hand="right"
config="src/specialist_sim/configs/fr3_omega.toml"
output_root="local/replay"

omega_root="$HOME/Documents/Omega-Infra-dev-fr3"
omega_python="$PWD/local/fr3-env/solver/bin/python"
pink_root="$HOME/Documents/pink"
pink_python="$PWD/local/fr3-env/pink/bin/python"

usage() {
  cat <<HELP
用法: FR3_replay.sh (--scipy | --omega-limited | --pink) [选项]
每次运行一个后端，默认生成 CSV、JSON 和 MP4。

  --dataset PATH       trial 目录（默认 $dataset）
  --session ID         不带 .rrd 的 session ID（默认 $session）
  --layer NAME         状态层（默认 $layer）
  --hand left|right    录制的手（默认 $hand）
  --config PATH        安装配置（默认 $config）
  --output-root PATH   输出根目录（默认 $output_root）
  --no-video           只生成 CSV 和 JSON
  -h, --help           显示用法
HELP
}

fail() { echo "FR3_replay.sh: $*" >&2; exit 2; }

solver=""
video_args=(--video)
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --scipy|--omega-limited|--pink)
      [[ -z "$solver" ]] || fail "每次请选择一个后端"
      solver=${1#--}
      ;;
    --dataset|--session|--layer|--hand|--config|--output-root)
      [[ -n "${2:-}" && "$2" != --* ]] || fail "$1 需要一个值"
      case "$1" in
        --dataset) dataset=$2 ;;
        --session) session=$2 ;;
        --layer) layer=$2 ;;
        --hand) hand=$2 ;;
        --config) config=$2 ;;
        --output-root) output_root=$2 ;;
      esac
      shift
      ;;
    --no-video) video_args=() ;;
    *) fail "未知参数 $1，可用 --help 查看用法" ;;
  esac
  shift
done

[[ -n "$solver" ]] || fail "请选择 --scipy、--omega-limited 或 --pink"
[[ "$hand" == left || "$hand" == right ]] || fail "--hand 需要 left 或 right"
[[ "$session" != */* && "$session" != . && "$session" != .. ]] || fail "--session 需要 session ID"

backend_args=()
case "$solver" in
  omega-limited) backend_args=(--omega-root "$omega_root" --omega-python "$omega_python") ;;
  pink) backend_args=(--pink-root "$pink_root" --pink-python "$pink_python") ;;
esac

replay_dir="$output_root/$session/$hand"
mkdir -p -- "$replay_dir"
out=$(mktemp -d -- "$replay_dir/$solver.XXXXXX")
echo "输出目录: $out"
export MUJOCO_GL=${MUJOCO_GL:-egl}
exec .venv/bin/sim-replay-arm --solver "$solver" "${backend_args[@]}" \
  --dataset "$dataset" --session "$session" --layer "$layer" --hand "$hand" \
  --config "$config" --output-dir "$out" "${video_args[@]}"
