import copy
import datetime
import json
import logging


import networkx as nx
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class DeviceInfoGenerator:
    def __init__(
        self,
        device_id: str,
        basis_gates_1q: list[str],
        basis_gates_2q: list[str],
        small_rows: int = 2,
        small_cols: int = 2,
        large_rows: int = 4,
        large_cols: int = 4,
        qubit_index_list: list[int] | str = "tool/qubit_index_list.csv",
        system_note: dict | str = "tool/.system_note.json",
    ):
        """
        初期化時に各種パラメータやファイルパス、もしくはデータ自体を設定する。

        Args:
            device_id (str): デバイスID
            basis_gates_1q (list[str]): シングルキュービットゲートのリスト
            basis_gates_2q (list[str]): 2キュービットゲートのリスト
            small_rows (int): 小さなラティスの行数
            small_cols (int): 小さなラティスの列数
            large_rows (int): 大きなラティスのタイル数（行方向）
            large_cols (int): 大きなラティスのタイル数（列方向）
            qubit_index_list (list[int] or str): 利用可能な物理キュービットインデックスのリスト、もしくはそのファイルパス
            system_note (dict or str): キャリブレーション情報の辞書、もしくはそのファイルパス
        """
        self.device_id = device_id
        self.basis_gates_1q = basis_gates_1q
        self.basis_gates_2q = basis_gates_2q
        self.small_rows = small_rows
        self.small_cols = small_cols
        self.large_rows = large_rows
        self.large_cols = large_cols
        self.qubit_index_list = qubit_index_list
        self.system_note = system_note

        self.mapping: dict[tuple[int, int], int] = {}
        self.pos: dict[int, tuple[int, int]] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self.available_qubit_indices: list[int] = []

    def load_qubit_index_list(self) -> None:
        """
        利用可能な物理キュービットインデックスを読み込む。
        qubit_index_listがリストの場合はそのまま使用し、
        ファイルパスの場合はCSVファイルから読み込む。
        """
        if isinstance(self.qubit_index_list, list):
            self.available_qubit_indices = self.qubit_index_list
        elif isinstance(self.qubit_index_list, str):
            try:
                with open(self.qubit_index_list, "r") as f:
                    content = f.read().strip()
                self.available_qubit_indices = [int(x) for x in content.split(",")]
            except Exception as e:
                logger.error(f"Failed to load qubit index list: {e}")
                raise
        else:
            raise ValueError("qubit_index_list must be a list of ints or a file path string.")

    def load_system_note(self) -> None:
        """
        キャリブレーション情報を読み込む。
        system_noteが辞書の場合はそのまま使用し、
        ファイルパスの場合はJSONファイルから読み込む。
        """
        if isinstance(self.system_note, dict):
            return  # すでに辞書形式なら何もしない
        elif isinstance(self.system_note, str):
            try:
                with open(self.system_note, "r") as f:
                    self.system_note = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load system note: {e}")
                raise
        else:
            raise ValueError("system_note must be a dict or a file path string.")

    def generate_mapping_and_pos(self) -> None:
        """
        小さなラティス（small_rows x small_cols）を大きなラティス（large_rows x large_cols）にタイル状に配置し、
        グリッド座標からキュービットインデックスへのマッピングと可視化用の座標を生成する。
        """
        total_rows = self.small_rows * self.large_rows
        total_cols = self.small_cols * self.large_cols
        small_size = self.small_rows * self.small_cols

        for global_row in range(total_rows):
            for global_col in range(total_cols):
                # 大・小のラティス内での位置を算出
                large_row, small_row = divmod(global_row, self.small_rows)
                large_col, small_col = divmod(global_col, self.small_cols)

                # 大ラティスと小ラティス内でのインデックスを計算
                large_index = large_row * self.large_cols + large_col
                small_index = small_row * self.small_cols + small_col
                qubit_index = large_index * small_size + small_index

                self.mapping[(global_row, global_col)] = qubit_index
                # 可視化用の座標（x: 列、y: -行）
                self.pos[qubit_index] = (global_col, -global_row)

    def generate_full_topology(self) -> None:
        """
        8x8のグリッドグラフを作成し、上記のマッピングを適用して物理キュービットのグラフを生成する。
        その後、有向グラフへと変換する。
        """
        grid = nx.grid_2d_graph(8, 8)

        if not self.mapping or not self.pos:
            self.generate_mapping_and_pos()

        mapped_grid = nx.relabel_nodes(grid, self.mapping)

        directed_graph = nx.DiGraph()
        for u, v in mapped_grid.edges():
            directed_graph.add_edge(u, v)
            # 双方向にしたい場合は以下を有効にする:
            # directed_graph.add_edge(v, u)
        self.graph = directed_graph

    def map_physical_to_virtual(self, physical_index: int) -> int:
        """
        物理キュービットインデックスから仮想キュービットインデックスへマッピングする。

        Raises:
            ValueError: 利用可能なリストに物理インデックスが存在しない場合
        """
        if not self.available_qubit_indices:
            self.load_qubit_index_list()
        if physical_index not in self.available_qubit_indices:
            raise ValueError(
                f"Physical qubit index {physical_index} is not available. Available: {self.available_qubit_indices}"
            )
        virtual_index = self.available_qubit_indices.index(physical_index)
        logger.debug(f"Mapped physical qubit {physical_index} to virtual qubit {virtual_index}")
        return virtual_index

    def map_virtual_to_physical(self, virtual_index: int) -> int:
        """
        仮想キュービットインデックスから物理キュービットインデックスへマッピングする。

        Raises:
            ValueError: 仮想インデックスが範囲外の場合
        """
        if not self.available_qubit_indices:
            self.load_qubit_index_list()
        if virtual_index < 0 or virtual_index >= len(self.available_qubit_indices):
            raise ValueError(
                f"Virtual qubit index {virtual_index} is out of range (max {len(self.available_qubit_indices) - 1})."
            )
        physical_index = self.available_qubit_indices[virtual_index]
        logger.debug(f"Mapped virtual qubit {virtual_index} to physical qubit {physical_index}")
        return physical_index

    def relabel_graph_physical_to_virtual(self) -> None:
        """
        グラフのノードを物理インデックスから仮想インデックスへリラベルする。
        利用可能な物理デバイスに存在しないノードは削除する。
        """
        mapping = {}
        new_pos = {}
        nodes_to_remove = []

        for physical_node in list(self.graph.nodes()):
            try:
                virtual_node = self.map_physical_to_virtual(physical_node)
                mapping[physical_node] = virtual_node
                new_pos[virtual_node] = self.pos[physical_node]
            except ValueError:
                logger.info(f"Physical node {physical_node} not available in the real machine.")
                nodes_to_remove.append(physical_node)

        pruned_graph = copy.deepcopy(self.graph)
        pruned_graph.remove_nodes_from(nodes_to_remove)

        self.graph = nx.relabel_nodes(pruned_graph, mapping)
        self.pos = new_pos

    def set_qubit_and_edge_properties(self) -> None:
        """
        ノード（キュービット）とエッジ（カップリング）に対して各種プロパティを設定する。
        キャリブレーション情報が利用できない場合はデフォルト値が設定される。
        """
        self.load_system_note()

        calibrated_qubits = list(self.system_note.get("average_gate_fidelity", {}).keys())
        for virtual_node in list(self.graph.nodes()):
            physical_node = self.map_virtual_to_physical(virtual_node)
            if virtual_node not in calibrated_qubits:
                logger.info(f"Qubit {virtual_node} is not calibrated.")
                node_data = {
                    "id": virtual_node,
                    "physical_id": physical_node,
                    "position": {"x": self.pos[virtual_node][0], "y": self.pos[virtual_node][1]},
                    "fidelity": 0.0,
                    "meas_error": {"prob_meas1_prep0": 0.0, "prob_meas0_prep1": 0.0},
                    "qubit_lifetime": {"t1": 0.0, "t2": 0.0},
                    "gate_duration": {
                        gate: 0 for gate in ("rz", "sx", "x") if gate in self.basis_gates_1q
                    },
                }
            else:
                node_data = {
                    "id": virtual_node,
                    "physical_id": physical_node,
                    "position": {"x": self.pos[virtual_node][0], "y": self.pos[virtual_node][1]},
                    "fidelity": self.system_note["average_gate_fidelity"].get(f"Q{physical_node:02}", 0.0),
                    "meas_error": {
                        "prob_meas1_prep0": self.system_note["readout"].get(f"Q{physical_node:02}", {}).get("prob_meas1_prep0", 0.0),
                        "prob_meas0_prep1": self.system_note["readout"].get(f"Q{physical_node:02}", {}).get("prob_meas0_prep1", 0.0),
                        "readout_assignment_error": self.system_note["readout"].get(f"Q{physical_node:02}", {}).get("readout_assignment_error", 0.0),
                    },
                    "qubit_lifetime": {
                        "t1": self.system_note["t1"].get(f"Q{physical_node:02}", 0.0),
                        "t2": self.system_note["t2"].get(f"Q{physical_node:02}", 0.0),
                    },
                    "gate_duration": {
                        gate: 0 for gate in ("rz", "sx", "x") if gate in self.basis_gates_1q
                    },
                }
            self.graph.nodes[virtual_node].update(node_data)

        calibrated_edges = []
        for key in self.system_note.get("cr_params", {}).keys():
            parts = key.split("-")
            if len(parts) == 2:
                calibrated_edges.append((parts[0], parts[1]))

        for u, v in list(self.graph.edges()):
            physical_u = self.map_virtual_to_physical(u)
            physical_v = self.map_virtual_to_physical(v)
            if (f"Q{physical_u:02}", f"Q{physical_v:02}") in calibrated_edges:
                edge_data = {
                    "control": u,
                    "target": v,
                    "fidelity": 0.0,
                    "gate_duration": {
                        gate: 0 for gate in ("cx", "rzx90") if gate in self.basis_gates_2q
                    },
                }
            else:
                logger.info(f"Coupling {u}-{v} is not calibrated.")
                edge_data = {
                    "control": u,
                    "target": v,
                    "fidelity": 0.0,
                    "gate_duration": {
                        gate: 0 for gate in ("cx", "rzx90") if gate in self.basis_gates_2q
                    },
                }
            self.graph.edges[u, v].update(edge_data)

    def dump_topology_json(self, output_json_file: str, indent: int = 2) -> None:
        """
        デバイスのトポロジーをJSON形式で出力する。
        """
        topology = {
            "name": "test_device",
            "device_id": self.device_id,
            "qubits": sorted(
                [data for _, data in self.graph.nodes(data=True)], key=lambda d: d["id"]
            ),
            "couplings": sorted(
                [data for _, _, data in self.graph.edges(data=True)], key=lambda d: d["control"]
            ),
            "timestamp": str(datetime.datetime.now()),
        }
        json_output = json.dumps(topology, indent=indent)
        try:
            with open(output_json_file, "w") as f:
                f.write(json_output)
        except Exception as e:
            logger.error(f"Failed to dump topology JSON: {e}")
            raise
    
    def dump_topology_dict(self) -> dict:
        """
		デバイスのトポロジーを辞書形式で返す。
		"""
        topology = {
            "name": "test_device",
			"device_id": self.device_id,
			"qubits": sorted(
				[data for _, data in self.graph.nodes(data=True)], key=lambda d: d["id"]
			),
			"couplings": sorted(
				[data for _, _, data in self.graph.edges(data=True)], key=lambda d: d["control"]
			),
			"timestamp": str(datetime.datetime.now()),
		}
        return topology

    def dump_topology_png(self, output_png_file: str, figsize: tuple[int, int] = (5, 5)) -> None:
        """
        デバイスのトポロジーをPNG画像として出力する。
        """
        plt.figure(figsize=figsize)
        nx.draw(
            self.graph,
            pos=self.pos,
            with_labels=True,
            node_color="white",
            edge_color="black",
            font_color="black",
            arrowsize=14,
        )
        plt.savefig(output_png_file)
        plt.close()
    from typing import Optional
    def generate_device_topology(self, output_json_file: Optional[str] = None, output_png_file: Optional[str] = None, save=False) -> None:
        """
        トポロジー生成の全工程を実行し、JSONおよびPNGで出力する。

        1. 物理キュービットインデックスの読み込み
        2. 8x8グリッドからのトポロジー生成
        3. 物理→仮想へのリラベル
        4. キャリブレーション情報の適用によるプロパティ設定
        5. JSON/PNGへの出力
        """
        self.load_qubit_index_list()
        self.generate_full_topology()
        self.relabel_graph_physical_to_virtual()
        self.load_system_note()
        self.set_qubit_and_edge_properties()
        if save:
            if output_json_file:
                self.dump_topology_json(output_json_file)
            if output_png_file:
                self.dump_topology_png(output_png_file)


if __name__ == "__main__":
    try:
        generator = DeviceInfoGenerator(
            device_id="1",
            basis_gates_1q=["rz", "sx", "x"],
            basis_gates_2q=["cx", "rzx90"],
        )
        generator.generate_device_topology(
            output_json_file="config/_device_topology.json",
            output_png_file="config/_device_topology.png",
        )
    except Exception:
        logger.error("Exception occurred during topology generation.", exc_info=True)
