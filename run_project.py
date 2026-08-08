"""IDE one-click entry for the complete product recommendation project."""

from __future__ import annotations

try:
    from .retailrocket_main import run_pipeline
except ImportError:
    from retailrocket_main import run_pipeline


# ======================== 只调整下面 4 项主要参数 ========================
# 是否复用已经训练完成的完整产物。
# True：存在完整模型包时直接复用，不重复训练；False：重新初始化并执行全流程。
REUSE_EXISTING_MODEL = True

# 是否使用快速检查模式。
# True：只读取部分数据并减少调优组合，用于确认流程能运行；False：使用全量数据正式训练。
QUICK_MODE = False

# 每个会话最终返回的推荐商品数，可填 1~20。
# 数值越大，推荐列表越长，Top-K 覆盖可能提高，但批量输出也会变大。
TOP_K = 10

# 每个会话最多保留的候选商品数，可填 10~200，且不能小于 TOP_K。
# 数值越大，候选召回可能提高，但特征生成、排序训练和内存占用都会增加。
MAX_CANDIDATES = 50


if __name__ == "__main__":
    run_pipeline(
        top_k=TOP_K,
        max_candidates=MAX_CANDIDATES,
        quick=QUICK_MODE,
        reuse_existing=REUSE_EXISTING_MODEL,
    )
