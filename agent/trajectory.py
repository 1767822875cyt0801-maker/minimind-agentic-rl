
#记录 Agent 与工具交互轨迹的数据结构和保存函数import json
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

#单步轨迹记录
@dataclass
class TrajectoryStep:
    role: str
    type: str
    content: str
    #附加元数据字典，用于存储额外信息
    meta: Dict[str, Any] = field(default_factory=dict)

#完整轨迹（一次任务）
@dataclass
class ToolTrajectory:
    task_id: str
    prompt: str
    tools: List[str]
    steps: List[TrajectoryStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    done: bool = False
    reward: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def trajectory_to_dict(traj: ToolTrajectory) -> Dict[str, Any]:
    return asdict(traj)

#将 ToolTrajectory 对象列表写入一个 JSONL（JSON Lines）格式的文件中
def save_trajectory_jsonl(path: str, trajectories: List[ToolTrajectory]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as f:
        for traj in trajectories:
            f.write(json.dumps(trajectory_to_dict(traj), ensure_ascii=False) + "\n")

