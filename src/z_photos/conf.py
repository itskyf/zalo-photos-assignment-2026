from dataclasses import dataclass, field
from pathlib import Path

import accelerate


@dataclass
class Config:
    data_root: Path
    model_name_or_path: str
    seed: int

    bronze_dir: Path = field(init=False)
    silver_dir: Path = field(init=False)
    gold_dir: Path = field(init=False)

    def __post_init__(self):
        """Tự động thiết lập đường dẫn và khởi tạo thư mục ngay khi Config được
        tạo."""

        # Thiết lập base cho 3 lớp Medallion
        self.bronze_dir = self.data_root / "bronze"
        self.silver_dir = self.data_root / "silver"
        self.gold_dir = self.data_root / "gold"

        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.gold_dir.mkdir(parents=True, exist_ok=True)

        accelerate.utils.set_seed(self.seed)


cfg = Config(
    data_root=Path("../data"),
    model_name_or_path="google/siglip2-so400m-patch16-384",
    seed=42,
)
