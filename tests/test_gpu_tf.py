import tensorflow as tf 


def gpuinfo(gpuidx):
    """
    Get GPU information

    Parameters
    ----------
    gpuidx : int
        GPU index

    Returns
    -------
    dict :
        GPU information in dictionary
    """
    import subprocess

    out_dict = {}
    try:
        sp = subprocess.Popen(
            ['nvidia-smi', '-q', '-i', str(gpuidx), '-d', 'MEMORY'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out_str = sp.communicate()
        out_list = out_str[0].decode("utf-8").split('BAR1', 1)[0].split('\n')
        for item in out_list:
            if ':' in item:
                fragments = item.split(':')
                if len(fragments) == 2:
                    out_dict[fragments[0].strip()] = fragments[1].strip()
    except Exception as e:
        print(e)
    return out_dict


def getfreegpumem(gpuidx):
    """
    Get free GPU memory

    Parameters
    ----------
    gpuidx : int
        GPU index

    Returns
    -------
    int :
        Free memory size
    """
    info = gpuinfo(gpuidx)
    if len(info) > 0:
        return int(info['Free'].replace('MiB', '').strip())
    else:
        return -1
    
def get_free():
    n_devices = len(tf.compat.v1.config.experimental.get_visible_devices())
    l = []
    for i in range(n_devices):
        l.append(getfreegpumem(i))
    return l

if __name__ == "__main__":
    print(f"Is GPU available: {tf.test.is_gpu_available()}")
    n_devices = len(tf.compat.v1.config.experimental.get_visible_devices())
    for i in range(n_devices):
        print(getfreegpumem(i))
    
    