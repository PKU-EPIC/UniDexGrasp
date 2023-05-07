import hashlib


def md5sum(filename, block_size=None):
    if block_size is None:
        block_size = 65536
    hash_res = hashlib.md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            hash_res.update(block)
    return hash_res.hexdigest()


def check_md5sum(filename, md5, block_size=None):
    if not (isinstance(md5, str) and len(md5) == 32):
        raise ValueError(f"MD5 must be 32 chars: {md5}")
    md5_actual = md5sum(filename, block_size=block_size)
    if md5_actual == md5:
        return True
    else:
        print(f"MD5 does not match!: {filename} has md5 {md5_actual}, target md5 is {md5}")
        return False
