import os
import sys
import re

# export all source files to a single file for botzone uploading

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

def process_files(c_files, export_filename):
    all_includes = set()
    all_code = []
    flag = False

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
                        flag = True
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
    
    with open(export_filename, 'w') as export_file:
        for include in sorted(all_includes):
            export_file.write(include + '\n')
        export_file.write('\n')
        for line in all_code:
            export_file.write(line)

    return flag

def main():
    if len(sys.argv) > 1:
        ext_file = sys.argv[1]
    else: 
        ext_file = 'botzone.c'

    # 获取 ./src/ 目录下的所有 .c 文件
    src_files = [os.path.join('src', f) for f in os.listdir('src') if f.endswith('.c')]
    # 添加 ./botzone.c 文件
    c_files = src_files + [ext_file]
    
    os.makedirs('export', exist_ok=True)

    export_file = os.path.join('export', ext_file)

    while process_files(c_files, export_file):
        c_files = [export_file]
    

if __name__ == "__main__":
    main()
