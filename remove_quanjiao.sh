# 替换全角句号。为半角句号.
find . -type f \( -name "*.cuh" -o -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -exec sed -i 's/。/./g' {} +

# 替换全角逗号，为半角逗号,
find . -type f \( -name "*.cuh" -o -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -exec sed -i 's/，/,/g' {} +

# 替换全角分号；为半角分号;
find . -type f \( -name "*.cuh" -o -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -exec sed -i 's/；/;/g' {} +

# 替换全角冒号：为半角冒号:
find . -type f \( -name "*.cuh" -o -name "*.cu" -o -name "*.h" -o -name "*.cpp" \) -exec sed -i 's/：/:/g' {} +

