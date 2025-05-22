import subprocess
import pandas as pd
import io
import datetime

# 加密文件路径
encrypted_file = "encrypted_file.enc"

# 加密密钥
password = "your_password"


def decrypt_file_to_pandas(encrypted_file: str, remove_file=True):
    try:
        # 解密后的文件路径
        decrypted_file = f"temp_decrypted_file{datetime.datetime.now().timestamp()}.csv"
        # 使用 openssl 解密文件
        openssl_command = f'openssl enc -d -aes-256-cbc -salt -pbkdf2 -in {encrypted_file} -out {decrypted_file} -k {password}'
        subprocess.run(openssl_command, shell=True, check=True)

        # 读取解密后的文件到 Pandas DataFrame
        df = pd.read_csv(decrypted_file)

        # print("文件解密完成，已加载到 Pandas DataFrame:")
        # print(df.head())
        if remove_file:
            import os
            os.remove(decrypted_file)
        return df

    except subprocess.CalledProcessError as e:
        print(f"解密过程中出现错误: {e}")
    except FileNotFoundError:
        print("未找到加密文件，请检查文件路径。")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return None
