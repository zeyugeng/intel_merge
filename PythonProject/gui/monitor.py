import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk


class BirdMonitorWindow:
    """鸟类监测 GUI 原型：展示声觉/视觉识别结果与视频区域。"""

    def __init__(self, root):
        self.root = root
        self.root.title("鸟类活动 AI 监测员")
        self.root.geometry("1000x820")
        self.root.minsize(900, 700)

        self.main_frame = tk.Frame(root, bg="#d9d9d9")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.top_frame = tk.LabelFrame(
            self.main_frame,
            text="鸟类信息列表",
            font=("微软雅黑", 10),
            bd=1,
            relief=tk.SOLID,
            bg="#d9d9d9",
        )
        self.top_frame.pack(fill=tk.BOTH, expand=False)
        self._create_table(self.top_frame)

        self.bottom_frame = tk.Frame(
            self.main_frame,
            bd=2,
            relief=tk.SOLID,
            bg="#d9d9d9",
            height=420,
        )
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.bottom_frame.pack_propagate(False)

        self.video_label = tk.Label(self.bottom_frame, bg="black", anchor="center")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        self.current_tk_image = None
        self.show_placeholder_image()

        self.control_frame = tk.Frame(self.main_frame, bg="#d9d9d9")
        self.control_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Button(
            self.control_frame,
            text="插入演示记录",
            font=("微软雅黑", 10),
            command=self.demo_add_record,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            self.control_frame,
            text="刷新演示画面",
            font=("微软雅黑", 10),
            command=self.show_demo_image,
        ).pack(side=tk.LEFT, padx=5)

        self._init_empty_rows()

    def _create_table(self, parent):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", rowheight=30, font=("微软雅黑", 10))
        style.configure("Treeview.Heading", font=("微软雅黑", 10, "bold"))

        columns = ("index", "audio_species", "position", "visual_species", "tracking")
        self.tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)
        self.tree.heading("index", text="序号")
        self.tree.heading("audio_species", text="鸟类种类（声音）")
        self.tree.heading("position", text="声源位置")
        self.tree.heading("visual_species", text="鸟类种类（视觉）")
        self.tree.heading("tracking", text="当前跟踪目标")

        self.tree.column("index", width=60, anchor="center")
        self.tree.column("audio_species", width=220, anchor="center")
        self.tree.column("position", width=140, anchor="center")
        self.tree.column("visual_species", width=220, anchor="center")
        self.tree.column("tracking", width=180, anchor="center")

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def _init_empty_rows(self):
        for i in range(1, 11):
            self.tree.insert("", tk.END, values=(i, "", "", "", ""))

    def add_record(self, audio_species: str, position: str, visual_species: str, tracking: str):
        rows = []
        for item in self.tree.get_children():
            row = self.tree.item(item, "values")
            if any(str(cell).strip() for cell in row[1:]):
                rows.append(row)

        rows.insert(0, ("", audio_species, position, visual_species, tracking))
        rows = rows[:10]

        for item in self.tree.get_children():
            self.tree.delete(item)

        for idx in range(10):
            if idx < len(rows):
                row = rows[idx]
                self.tree.insert("", tk.END, values=(idx + 1, row[1], row[2], row[3], row[4]))
            else:
                self.tree.insert("", tk.END, values=(idx + 1, "", "", "", ""))

    def show_placeholder_image(self):
        img = Image.new("RGB", (900, 380), "black")
        draw = ImageDraw.Draw(img)
        draw.text((360, 185), "视频显示区域", fill="white")
        self.update_video_image(img)

    def show_demo_image(self):
        img = Image.new("RGB", (900, 380), "black")
        draw = ImageDraw.Draw(img)
        draw.rectangle((90, 40, 250, 260), outline="lime", width=4)
        draw.rectangle((520, 80, 700, 300), outline="lime", width=4)
        draw.text((90, 18), "Bird #1", fill="lime")
        draw.text((520, 58), "Bird #2 (tracked)", fill=(255, 0, 255))
        self.update_video_image(img)

    def update_video_image(self, pil_image):
        frame_w = max(self.video_label.winfo_width(), 900)
        frame_h = max(self.video_label.winfo_height(), 380)
        img = pil_image.copy()
        img.thumbnail((frame_w, frame_h))
        self.current_tk_image = ImageTk.PhotoImage(img)
        self.video_label.config(image=self.current_tk_image)

    def demo_add_record(self):
        self.add_record(
            audio_species="Zosterops japonicus / 绣眼鸟",
            position="x=0.32",
            visual_species="bird (YOLO26)",
            tracking="Bird #1",
        )


def main():
    root = tk.Tk()
    BirdMonitorWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
