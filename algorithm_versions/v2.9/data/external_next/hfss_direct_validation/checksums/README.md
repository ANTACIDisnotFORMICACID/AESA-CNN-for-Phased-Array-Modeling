# 导出文件哈希

全部 HFSS 原始导出完成后，在 `hfss_direct_validation` 目录运行：

```powershell
Get-ChildItem -File -Recurse |
  Where-Object { $_.FullName -notmatch '\\checksums\\' } |
  Get-FileHash -Algorithm SHA256 |
  Export-Csv .\checksums\sha256.csv -NoTypeInformation -Encoding utf8
```

生成哈希后，原始导出进入只读状态；任何处理结果写到其他目录。
