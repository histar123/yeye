import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_auc_score, classification_report, roc_curve,
                             confusion_matrix)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib

# 中文显示配置
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ================== 核心模型部分 ==================
class ChurnPredictModel:
    def __init__(self):
        self.df = None
        self.pipeline = None
        self.categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        self.feature_names = self.categorical_features + self.numerical_features

        self.load_and_preprocess_data()
        self.train_model()

        final_result = self.evaluate_model()
        print("\n=== 模型评估结果 ===")
        print(f"准确率为{final_result["accuracy"]}")
        print(f"f1分数为{final_result["f1"]}")
        print(f"召回率为{final_result["recall"]}")
        print(f"auc为{final_result["roc_auc"]}")
        print(f"分类报告为\n{final_result["report"]}")

        self.vision()

    def load_and_preprocess_data(self):
        """数据加载与预处理"""
        data = fetch_openml(name="Telco-Customer-Churn", version=1, as_frame=True)
        self.df = data.frame.copy()

        # 转换 TotalCharges 为数值类型，并用 MonthlyCharges * tenure 填充缺失值
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        self.df['TotalCharges'] = self.df['TotalCharges'].fillna(
            self.df['MonthlyCharges'] * self.df['tenure']
        )

        # 清理分类特征的字符串
        for col in self.categorical_features:
            if self.df[col].dtype == 'object':  # 字符串类型在pandas中被标记为object类型
                self.df.loc[:, col] = self.df[col].str.strip(" '\"")  # 同时移除空白和引号

    def train_model(self):
        """模型训练"""
        X = self.df.drop(['Churn'], axis=1)
        y = self.df['Churn'].map({'Yes': 1, 'No': 0})

        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist'), self.categorical_features),
            ('num', 'passthrough', self.numerical_features)
        ])

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=250,
                max_depth=20,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.pipeline.fit(X_train, y_train)
        print(X_train)
        self.X_test, self.y_test = X_test, y_test

    def evaluate_model(self):
        """模型评估"""

        y_pred = self.pipeline.predict(self.X_test)
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

        # 保存模型
        joblib.dump(self.pipeline, 'churn_model.pkl')

        return {
            'accuracy': accuracy_score(self.y_test, y_pred),
            "f1":f1_score(self.y_test,y_pred),
            "recall":recall_score(self.y_test,y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_proba),
            'report': classification_report(self.y_test, y_pred)
        }
    def vision(self):
        # 可视化
        # 1.AUC曲线
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        y_pred = self.pipeline.predict(self.X_test)
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]

        fpr,tpr,thersholds = roc_curve(self.y_test,y_proba)
        auc = roc_auc_score(self.y_test,y_proba)
        plt.figure(figsize=(8,6))
        plt.plot(fpr,tpr,color = "darkorange",label = f"ROC曲线(AUC={auc:.2f})")
        plt.plot([0,1],[0,1],color = "black",linestyle = "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC曲线")
        plt.legend(loc = "lower right")
        plt.show()
        # 混淆矩阵
        cm = confusion_matrix(self.y_test,y_pred)
        sns.heatmap(cm,annot = True,fmt="d",cmap="Blues")
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.show()

    def predict_single(self, input_data):
        """预测单个客户"""
        df = pd.DataFrame([input_data])
        for num_feat in self.numerical_features:
            df[num_feat] = pd.to_numeric(df[num_feat])

        proba = self.pipeline.predict_proba(df)[0][1]
        prediction = 1 if proba > 0.5 else 0
        return proba, prediction

    def predict_batch(self, file_path):
        """批量预测CSV文件"""
        df = pd.read_csv(file_path)

        # 检查是否包含所有必要特征
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_cols)}")

        # 确保数据类型正确
        for num_feat in self.numerical_features:
            df[num_feat] = pd.to_numeric(df[num_feat])

        # 预测
        proba = self.pipeline.predict_proba(df)[:, 1]
        predictions = (proba > 0.5).astype(int)

        # 添加预测结果到数据框
        result_df = df.copy()
        result_df['流失概率'] = proba
        result_df['预测结果'] = np.where(predictions == 1, '高风险（流失）', '低风险（留存）')

        return result_df


# ================== GUI界面部分 ==================
class ChurnPredictorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("客户流失预测系统")
        self.master.geometry("1200x800")
        self.master.configure(bg='#f0f0f0')

        # 创建模型
        self.model = ChurnPredictModel()

        # 界面标签翻译字典
        self.label_translation = {
            'gender': '性别',
            'Partner': '配偶',
            'Dependents': '家属',
            'PhoneService': '电话服务',
            'MultipleLines': '多条电话服务',
            'InternetService': '网络服务',
            'OnlineSecurity': '网络安全服务',
            'OnlineBackup': '网络备份',
            'DeviceProtection': '设备保护',
            'TechSupport': '技术支持',
            'StreamingTV': '数字电视',
            'StreamingMovies': '数字电影',
            'Contract': '合约方式',
            'PaperlessBilling': '电子账单',
            'PaymentMethod': '支付方式',
            'SeniorCitizen': '是否是老年人',
            'tenure': '服务时长(月)',
            'MonthlyCharges': '月费用($)',
            'TotalCharges': '总费用($)'
        }

        # 创建主布局
        self.create_main_layout()

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪 | 模型准确率: {:.2%}".format(self.model.evaluate_model()['accuracy']))
        status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_main_layout(self):
        """创建主界面布局"""
        # 使用笔记本控件创建标签页
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建标签页
        self.create_single_prediction_tab()
        self.create_batch_prediction_tab()
        self.create_model_info_tab()

    def create_single_prediction_tab(self):
        """创建单个客户预测标签页"""
        self.single_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.single_tab, text='单个客户预测')

        # 创建主框架
        main_frame = ttk.Frame(self.single_tab)
        main_frame.pack(fill=tk.BOTH, expand = True, padx=15, pady=15)

        # 输入区域
        input_frame = ttk.LabelFrame(main_frame, text="客户信息输入", padding=10)
        input_frame.pack(fill=tk.X, pady=10)

        self.entries = {}
        features = self.model.categorical_features + self.model.numerical_features

        # 创建两列布局
        left_frame = ttk.Frame(input_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        right_frame = ttk.Frame(input_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 将特征分配到左右两列
        mid_point = len(features) // 2
        for idx, feat in enumerate(features):
            frame = left_frame if idx < mid_point else right_frame

            row_frame = ttk.Frame(frame)
            row_frame.pack(fill=tk.X, pady=3)

            # 使用中文标签
            ttk.Label(row_frame, text=f"{self.label_translation[feat]}:", width=20).pack(side=tk.LEFT)

            if feat in self.model.categorical_features:
                cb = ttk.Combobox(row_frame, values=self.model.df[feat].unique().tolist(), width=25)
                cb.set(self.model.df[feat].iloc[0])
                cb.pack(side=tk.LEFT, padx=5)
                self.entries[feat] = cb
            else:
                ent = ttk.Entry(row_frame, width=28)
                ent.insert(0, str(self.model.df[feat].iloc[0]))
                ent.pack(side=tk.LEFT, padx=5)
                self.entries[feat] = ent

        # 按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        predict_btn = ttk.Button(btn_frame, text="开始预测", command=self.predict_single, width=15)
        predict_btn.pack(side=tk.LEFT, padx=10)

        clear_btn = ttk.Button(btn_frame, text="清空输入", command=self.clear_single_input, width=15)
        clear_btn.pack(side=tk.LEFT, padx=10)

        # 结果展示区域
        result_frame = ttk.LabelFrame(main_frame, text="预测结果与建议", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 创建左右布局
        result_left = ttk.Frame(result_frame)
        result_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        result_right = ttk.Frame(result_frame)
        result_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 文本结果
        self.result_text = tk.Text(result_left, height=12, font=('微软雅黑', 10))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.config(state=tk.DISABLED)

        # 可视化
        self.viz_frame = ttk.Frame(result_right)
        self.viz_frame.pack(fill=tk.BOTH, expand=True)

        # 初始可视化
        self.create_initial_viz()

    def create_batch_prediction_tab(self):
        """创建批量预测标签页"""
        self.batch_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_tab, text='批量预测')

        # 主框架
        main_frame = ttk.Frame(self.batch_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # 上传区域
        upload_frame = ttk.LabelFrame(main_frame, text="文件上传", padding=10)
        upload_frame.pack(fill=tk.X, pady=10)

        ttk.Label(upload_frame, text="选择CSV文件:").pack(side=tk.LEFT, padx=5)

        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(upload_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        browse_btn = ttk.Button(upload_frame, text="浏览...", command=self.browse_file, width=10)
        browse_btn.pack(side=tk.LEFT, padx=5)

        predict_btn = ttk.Button(upload_frame, text="开始批量预测", command=self.predict_batch, width=15)
        predict_btn.pack(side=tk.RIGHT, padx=5)

        # 结果区域
        result_frame = ttk.LabelFrame(main_frame, text="预测结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 创建Treeview表格
        columns = ['客户ID'] + list(self.label_translation.values()) + ['流失概率', '预测结果']
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show='headings', height=15)

        # 设置列宽和标题
        col_widths = {
            '客户ID': 60,
            '性别': 60, '配偶': 60, '家属': 60, '电话服务': 80,
            '多条电话服务': 100, '网络服务': 80, '网络安全服务': 100, '网络备份': 80,
            '设备保护': 80, '技术支持': 80, '数字电视': 80, '数字电影': 80,
            '合约方式': 100, '电子账单': 80, '支付方式': 100, '老年人': 60,
            '服务时长(月)': 100, '月费用($)': 80, '总费用($)': 80,
            '流失概率': 80, '预测结果': 100
        }

        for col in columns:
            self.result_tree.heading(col, text=col)
            self.result_tree.column(col, width=col_widths.get(col, 100), anchor=tk.CENTER)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.pack(fill=tk.BOTH, expand=True)

        # 底部按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        save_btn = ttk.Button(btn_frame, text="保存结果", command=self.save_results, width=15)
        save_btn.pack(side=tk.RIGHT, padx=10)

        clear_btn = ttk.Button(btn_frame, text="清空表格", command=self.clear_tree, width=15)
        clear_btn.pack(side=tk.RIGHT, padx=10)

    def create_model_info_tab(self):
        """创建模型信息标签页"""
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text='模型信息')

        # 主框架
        main_frame = ttk.Frame(self.model_tab)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # 模型评估信息
        eval_frame = ttk.LabelFrame(main_frame, text="模型评估指标", padding=10)
        eval_frame.pack(fill=tk.X, pady=10)

        # 获取评估结果
        eval_results = self.model.evaluate_model()

        # 创建评估指标显示
        metrics_frame = ttk.Frame(eval_frame)
        metrics_frame.pack(fill=tk.X, pady=5)

        ttk.Label(metrics_frame, text=f"准确率: {eval_results['accuracy']:.2%}", font=('微软雅黑', 10, 'bold')).pack(
            side=tk.LEFT, padx=20)
        ttk.Label(metrics_frame, text=f"AUC-ROC: {eval_results['roc_auc']:.2%}", font=('微软雅黑', 10, 'bold')).pack(
            side=tk.LEFT, padx=20)

        # 分类报告
        report_frame = ttk.Frame(eval_frame)
        report_frame.pack(fill=tk.X, pady=5)

        report_text = tk.Text(report_frame, height=8, width=60)
        report_text.pack(fill=tk.BOTH, expand=True)
        report_text.insert(tk.END, eval_results['report'])
        report_text.config(state=tk.DISABLED)

        # 特征重要性
        feature_frame = ttk.LabelFrame(main_frame, text="特征重要性", padding=10)
        feature_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 获取特征重要性
        classifier = self.model.pipeline.named_steps['classifier']
        ohe_columns = self.model.pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
        all_features = list(ohe_columns) + self.model.numerical_features
        importances = classifier.feature_importances_

        # 取前15个重要特征
        sorted_idx = importances.argsort()[::-1][:15]
        top_features = np.array(all_features)[sorted_idx]
        top_importances = importances[sorted_idx]

        # 创建水平条形图
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_importances, align='center', color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()  # 最重要的特征显示在最上面
        for i, v in enumerate(top_importances):
            ax.text(v + 0.01, i, f"{v:.2f}", ha='left', va='center')
        ax.set_xlabel('重要性分数')
        ax.set_title('Top 15 重要特征')

        # 嵌入图表到GUI
        canvas = FigureCanvasTkAgg(fig, feature_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_initial_viz(self):
        """创建初始可视化"""
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        # 创建初始风险指示器
        ax.barh(['风险等级'], [0], color='#d3d3d3')
        ax.set_xlim(0, 1)
        ax.set_title('客户流失风险指示器')
        ax.set_xlabel('流失概率')
        ax.text(0.5, 0, "等待预测...", ha='center', va='center', fontsize=12, color='gray')

        # 嵌入图表到GUI
        self.canvas = FigureCanvasTkAgg(fig, self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def predict_single(self):
        """执行单个客户预测"""
        try:
            # 获取输入数据
            input_data = {feat: widget.get() for feat, widget in self.entries.items()}

            # 预测
            proba, prediction = self.model.predict_single(input_data)

            # 更新状态
            self.status_var.set(f"预测完成 | 流失概率: {proba:.2%}")

            # 显示结果
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)

            result = [
                "=== 预测结果 ===",
                f"流失概率: {proba:.2%}",
                f"风险等级: {'高风险（流失）' if prediction == 1 else '低风险（留存）'}",
                "\n=== 建议措施 ==="
            ]

            # 根据风险等级添加建议
            if proba > 0.7:
                result.append("▶ 提供专属优惠套餐")
                result.append("▶ 安排客户经理回访")
                result.append("▶ 优先处理服务请求")
                result.append("▶ 提供个性化服务方案")
            elif proba > 0.5:
                result.append("▶ 发送客户满意度问卷")
                result.append("▶ 提供短期优惠")
                result.append("▶ 推送相关服务信息")
                result.append("▶ 增强客户关系维护")
            else:
                result.append("▶ 推送最新服务信息")
                result.append("▶ 定期发送使用报告")
                result.append("▶ 提供增值服务推荐")
                result.append("▶ 维持良好客户关系")

            self.result_text.insert(tk.END, "\n".join(result))
            self.result_text.config(state=tk.DISABLED)

            # 更新可视化
            self.viz_frame.destroy()
            self.viz_frame = ttk.Frame(self.single_tab.winfo_children()[0].winfo_children()[2].winfo_children()[1])
            self.viz_frame.pack(fill=tk.BOTH, expand=True)

            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)

            # 创建风险指示器
            color = '#FF6B6B' if prediction == 1 else '#4ECDC4'
            ax.barh(['风险等级'], [proba], color=color)
            ax.set_xlim(0, 1)
            ax.set_title('客户流失风险指示器')
            ax.set_xlabel('流失概率')

            # 添加概率文本
            ax.text(proba / 2, 0, f"{proba:.1%}", ha='center', va='center',
                    fontsize=20, color='white', fontweight='bold')

            # 添加风险标签
            risk_label = "高风险" if prediction == 1 else "低风险"
            ax.text(proba + 0.05, 0, risk_label, ha='left', va='center',
                    fontsize=14, color=color, fontweight='bold')

            # 嵌入图表到GUI
            self.canvas = FigureCanvasTkAgg(fig, self.viz_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("预测错误", f"预测失败: {str(e)}")

    def browse_file(self):
        """浏览文件"""
        file_path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)

    def predict_batch(self):
        """执行批量预测"""
        file_path = self.file_path_var.get()
        if not file_path:
            messagebox.showwarning("警告", "请先选择CSV文件")
            return

        try:
            # 清空现有表格
            self.clear_tree()

            # 执行批量预测
            result_df = self.model.predict_batch(file_path)

            # 更新状态
            self.status_var.set(f"批量预测完成 | 共处理 {len(result_df)} 条记录")

            # 插入数据到表格
            for i, row in result_df.iterrows():
                # 创建客户ID
                customer_id = f"C{1000 + i}"

                # 获取特征值
                values = [customer_id] + [str(row[col]) for col in self.model.feature_names]

                # 添加预测结果
                values.append(f"{row['流失概率']:.4f}")
                values.append(row['预测结果'])

                self.result_tree.insert("", tk.END, values=values)

            # 保存结果到临时变量
            self.batch_results = result_df

            messagebox.showinfo("成功", f"批量预测完成！共处理 {len(result_df)} 条记录")

        except Exception as e:
            messagebox.showerror("批量预测错误", f"处理CSV文件失败: {str(e)}")

    def save_results(self):
        """保存预测结果"""
        if not hasattr(self, 'batch_results') or self.batch_results.empty:
            messagebox.showwarning("警告", "没有可保存的结果")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv")]
        )

        if save_path:
            try:
                self.batch_results.to_csv(save_path, index=False)
                messagebox.showinfo("成功", f"结果已保存至:\n{save_path}")
                self.status_var.set(f"结果已保存至: {save_path}")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存文件失败: {str(e)}")

    def clear_single_input(self):
        """清空单个客户输入"""
        for widget in self.entries.values():
            if isinstance(widget, ttk.Combobox):
                widget.set('')
            else:
                widget.delete(0, tk.END)

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state=tk.DISABLED)

        # 重置可视化
        self.viz_frame.destroy()
        self.viz_frame = ttk.Frame(self.single_tab.winfo_children()[0].winfo_children()[2].winfo_children()[1])
        self.viz_frame.pack(fill=tk.BOTH, expand=True)
        self.create_initial_viz()

        self.status_var.set("输入已清空")

    def clear_tree(self):
        """清空结果表格"""
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        if hasattr(self, 'batch_results'):
            del self.batch_results


if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnPredictorApp(root)
    root.mainloop()