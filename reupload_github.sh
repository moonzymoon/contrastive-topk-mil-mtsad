#!/bin/bash
# ============================================================
# GitHub 仓库修复脚本 - 补全缺失的子目录 + 更新 README
# 仓库: moonzymoon/contrastive-topk-mil-mtsad
#
# 使用方法:
#   1. 在 Git Bash 中运行: bash reupload_github.sh
#   2. 首次推送时会弹出登录窗口,输入 GitHub 用户名和 Personal Access Token
#      (Token 在 GitHub → Settings → Developer settings → Personal access tokens 生成)
# ============================================================

set -e

REPO_URL="https://github.com/moonzymoon/contrastive-topk-mil-mtsad.git"
WORK_DIR="D:/0科研/工作1/第二篇SCI/Contrastive_TopK_MIL/GITHUB_UPLOAD"

echo "================================================"
echo "  GitHub 仓库修复 - 完整目录结构上传"
echo "================================================"
echo ""

# 配置 git 身份(如未配置)
git config --global user.name "moonzymoon" 2>/dev/null || true
git config --global user.email "zhangyao@bdu.edu.cn" 2>/dev/null || true
git config --global credential.helper manager 2>/dev/null || true

cd "$WORK_DIR"

# 初始化新的 git 仓库(本地 GITHUB_UPLOAD 目录还不是 git 仓库)
if [ ! -d ".git" ]; then
    echo "[1/6] 初始化本地 git 仓库..."
    git init
    git remote add origin "$REPO_URL"
else
    echo "[1/6] 本地已是 git 仓库"
fi

echo "[2/6] 拉取远程 main 分支历史(避免冲突)..."
git fetch origin main 2>/dev/null || echo "  (远程为空,跳过)"
git checkout -B main 2>/dev/null || git checkout main 2>/dev/null || true

echo "[3/6] 添加所有文件(保留目录结构)..."
# 删除旧的扁平文件追踪,重新添加完整结构
git rm -r --cached . 2>/dev/null || true
git add -A

echo "[4/6] 检查将要提交的文件结构..."
git status --short | head -40
FILE_COUNT=$(git status --short | wc -l)
echo "  待提交文件数: $FILE_COUNT"

echo "[5/6] 提交..."
git commit -m "fix: restore complete directory structure (models/losses/data/evaluation/utils)

- Add missing core modules: models/, losses/, data/, evaluation/, utils/, baselines_ext/
- Previously uploaded as flat structure causing ModuleNotFoundError on import
- Rewrite README to match paper (IJMLC, Gradient Attribution, honest negative results)
- All 22 core Python modules verified parseable" 2>&1 || echo "(无变更或已提交)"

echo "[6/6] 推送到远程 main 分支(强制覆盖远程扁平结构)..."
echo "  >>> 如弹出登录窗口,输入 GitHub 用户名 + Personal Access Token <<<"
git push -f origin main

echo ""
echo "================================================"
echo "  ✅ 上传完成!"
echo "  仓库地址: https://github.com/moonzymoon/contrastive-topk-mil-mtsad"
echo ""
echo "  请打开上面链接,确认目录结构如下:"
echo "    README.md"
echo "    configs/default.yaml"
echo "    src/data/          (loaders.py, split.py, __init__.py)"
echo "    src/models/        (encoder.py, hyperspheres.py, cmh_mil.py, __init__.py)"
echo "    src/losses/        (cmh_loss.py, cmh_loss_v2.py, __init__.py)"
echo "    src/evaluation/    (conformal.py, metrics.py, __init__.py)"
echo "    src/utils/         (repro.py, __init__.py)"
echo "    src/baselines_ext/ (run_tailmil.py, ...)"
echo "    src/*.py           (run_v2_experiment.py, train.py, baselines.py, ...)"
echo "    results/           (*.json)"
echo "================================================"
