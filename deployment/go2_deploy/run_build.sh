#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${project_root}/deploy/build"
output_dir="${project_root}/output"

cmake -E remove_directory "${build_dir}"
cmake -S "${project_root}/deploy" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel

cmake -E remove_directory "${output_dir}"
cmake -E make_directory "${output_dir}"
cmake -E copy "${build_dir}/deploy" "${output_dir}/deploy"
cmake -E create_symlink "${project_root}/model" "${output_dir}/model"
cmake -E copy "${project_root}/deploy/config.json" "${output_dir}/config.json"
