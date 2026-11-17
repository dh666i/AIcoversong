# Contributing / 贡献指南

欢迎通过 Issue 描述问题，或以小而完整的 Pull Request 改进 AIcoversong。请先阅读 README、模型许可说明与安全政策。

## 开发边界

- 保留现有录音、模型与本机配置；不要为了测试覆盖用户数据。
- 音频效果改动应说明输入条件、参数差别和验证局限。用合成音或可公开授权素材，不能把一次数值变好描述为所有歌曲都更好听。
- UI 改动需要检查键盘焦点、窄屏、减少动效偏好、成功/失败/取消以及连续运行两次的状态。
- `Applio/` 是固定快照，修改上游文件时保留版权声明，并在 PR 中解释偏离上游的原因。

## 本地检查

```powershell
python tools/verify_package.py --tracked
python -m unittest discover -s tests -v
python -m compileall -q tools Applio
```

无需私人音色或 GPU 的测试应保持轻量。完整推理测试请单独说明显卡、输入是否可分享和实际运行的检查；不要向 CI 上传个人音色、歌曲或录音。

## 提交

提交前检查 `git diff --cached`。不要使用 `git add -f` 绕过个人数据排除。PR 描述应包括：具体问题、修复后的行为、验证方法、尚未验证的项目。欢迎提供中文说明，英文也可以。
