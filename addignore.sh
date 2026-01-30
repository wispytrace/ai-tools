# 1. 查找大于 100MB 的文件
# 2. 去掉前面的 ./ 符号
# 3. 追加到 .gitignore 文件中
find . -size +100M | sed 's|^\./||g' >> .gitignore

# 4. (可选) 打印出来看看忽略了什么
cat .gitignore