import os
import sys
import re

def collect_includes_and_code(file_path):
    includes = []
    code = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip().startswith('#include'):
                includes.append(line.strip())
            else:
                code.append(line)
    return includes, code

def process_header_file(header_file):
    content = []
    with open(header_file, 'r') as file:
        for line in file:
            if not line.strip().startswith('#pragma once'):
                content.append(line)
    return content

def main():
    if len(sys.argv) > 1:
        c_files = sys.argv[1:]
    else: 
    # 获取 ./src/ 目录下的所有 .c 文件
        src_files = [os.path.join('src', f) for f in os.listdir('src') if f.endswith('.c')]
        # 添加 ./botzone.c 文件
        c_files = src_files + ['botzone.c']
    
    all_includes = set()
    all_code = []

    for c_file in c_files:
        if os.path.exists(c_file):
            print(f"Processing {c_file}")
            includes, code = collect_includes_and_code(c_file)
            for include in includes:
                match = re.match(r'#include\s+"([^"]+)"', include)
                if match:
                    header_file = os.path.join('include', match.group(1))
                    if os.path.exists(header_file):
                        print(f"Expanding header file {header_file}")
                        header_content = process_header_file(header_file)
                        all_code.extend(header_content)
                        all_code.extend('\n')
                    else:
                        print(f"Header file {header_file} does not exist")
                else:
                    all_includes.add(include)
            all_code.extend(code)
            all_code.extend('\n')
        else:
            print(f"File {c_file} does not exist")
    
    os.makedirs('export', exist_ok=True)

    with open('export/export.c', 'w') as export_file:
        for include in sorted(all_includes):
            export_file.write(include + '\n')
        export_file.write('\n')
        for line in all_code:
            export_file.write(line)

if __name__ == "__main__":
    main()