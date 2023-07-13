import subprocess


if __name__ == "__main__":
    
    gpuidx = 0
    
    sp = subprocess.Popen(
            ['nvidia-smi', '-q', '-i', str(gpuidx), '-d', 'MEMORY,UTILIZATION'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    out_dict = {}
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split('\n')
    
    for item in out_list:
            if ':' in item:
                fragments = item.split(':')
                if len(fragments) == 2:
                    out_dict[fragments[0].strip()] = fragments[1].strip()
    
    print(out_dict)