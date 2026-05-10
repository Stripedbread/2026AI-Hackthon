# 🚀 黑客松项目初始化指南

## 前置准备(已完成)

- [x] GitHub CLI 已安装并登录：`gh auth status`
- [x] Python 3.11+ 已安装
- [x] Gradio 已安装：`pip install gradio`
- [x] ModelScope 账号已注册：https://modelscope.cn

---

## Step 1：创建 GitHub 仓库

```bash
gh repo create 2026AI-Hackthon --public --description "2026 AI Hackathon project" --clone
```

✅ 完成后仓库地址：`https://github.com/<用户名>/2026AI-Hackthon`

---

## Step 2：开发 Gradio 应用

### 2.1 创建 `app.py`

```python
import gradio as gr

def greet(name):
    return f"👋 你好 {name}！欢迎参加黑客松！"

with gr.Blocks(title="🚀 黑客松应用") as demo:
    gr.Markdown("# 🚀 黑客松演示应用")
    
    name_input = gr.Textbox(label="你的名字", placeholder="输入你的名字...")
    greet_btn = gr.Button("🎤 生成问候", variant="primary")
    greet_output = gr.Markdown()
    
    greet_btn.click(fn=greet, inputs=name_input, outputs=greet_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
```

### 2.2 创建 `requirements.txt`

```
gradio>=4.0.0
```

### 2.3 创建 `.gitignore`

```
__pycache__/
*.pyc
.venv/
.vscode/
*.egg-info/
```

### 2.4 本地测试

```bash
python app.py
# 访问 http://localhost:7860
```

✅ 确认功能正常后进入下一步。

---

## Step 3：推送到 GitHub

```bash
git add .
git commit -m "Initial commit: Gradio hackathon app"
git push -u origin master
```

---

## Step 4：部署到 ModelScope 创空间（获得公网 URL）

### 4.1 创建创空间

1. 访问 https://modelscope.cn/studios
2. 点击「创建创空间」
3. 填写信息：
   - **空间名称**：自定义（如 `demo`）
   - **SDK**：选择 **Gradio**
   - **可见性**：选择 **公开**
4. 创建后获得克隆地址

### 4.2 克隆创空间仓库

```bash
git lfs install
git clone http://oauth2:ms-b992cd79-197b-42f7-9c1b-d14c0ed0f9b2@www.modelscope.cn/studios/CapooCat123/AI-Hackthon.git
```

### 4.3 创建第一个 Gradio 应用 `app.py`

```python
import gradio as gr

def modelscope_quickstart(name):
    return "Welcome to modelscope, " + name + "!!"

demo = gr.Interface(fn=modelscope_quickstart, inputs="text", outputs="text")
demo.launch()
```

### 4.4 提交并推送部署

```bash
git add app.py
git commit -m "Add application file"
git push
```

✅ 推送后 ModelScope 自动构建，构建完成后公网 URL 即生效。

---

## 最终交付物

| 项目 | 链接 |
|------|------|
| GitHub 仓库 | `https://github.com/<用户名>/2026AI-Hackthon` |
| 公网部署 URL | `https://www.modelscope.cn/studios/CapooCat123/AI-Hackthon` |

---

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 检查 gh 登录 | `gh auth status` |
| 创建仓库 | `gh repo create <name> --public --clone` |
| 查看仓库状态 | `git status` |
| 提交推送 | `git add . && git commit -m "msg" && git push` |
| 安装依赖 | `pip install -r requirements.txt` |
| 本地运行 | `python app.py` |

# 为提高开发效率，全程不使用虚拟环境
