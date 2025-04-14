import boto3
import os

# 设置HTTP代理（如果您使用VPN）
os.environ['HTTP_PROXY'] = 'http://localhost:7890'
os.environ['HTTPS_PROXY'] = 'http://localhost:7890'



# S3资源
s3 = boto3.client(
    's3',
    aws_access_key_id=credentials['accessKeyId'],
    aws_secret_access_key=credentials['secretAccessKey'],
    aws_session_token=credentials['sessionToken']
)

# 读取S3 URLs的文本文件
file_path = r'E:\地理所\工作\徐州卫星同化_2025.2\OCO3卫星数据\subset_OCO3_L2_Lite_FP_11r_20250218_131202_S3URLs.txt'

# 指定保存文件的本地目录
save_directory = r'E:\地理所\工作\徐州卫星同化_2025.2\OCO3卫星数据'

# 确保保存目录存在
os.makedirs(save_directory, exist_ok=True)

try:
    with open(file_path, 'r', encoding='utf-8') as file:
        s3_urls = file.readlines()

    # 去除每行末尾的换行符
    s3_urls = [url.strip() for url in s3_urls]

    # 打印读取的S3 URLs
    print("读取到的S3 URLs:")
    print(s3_urls)

    # 下载文件函数
    def download_file(bucket_name, object_name, file_name):
        """
        从S3下载文件到本地。
        :param bucket_name: S3 bucket的名称
        :param object_name: S3 bucket中的对象名称
        :param file_name: 本地保存的文件名
        :return: None
        """
        try:
            s3.download_file(bucket_name, object_name, file_name)
            print(f"文件 {object_name} 已成功下载到 {file_name}")
        except Exception as e:
            print(f"下载 {object_name} 失败: {e}")

    # 示例：下载文件
    bucket_name = 'gesdisc-cumulus-prod-protected'

    for url in s3_urls:
        object_name = url.replace("s3://", "").strip()  # 去掉s3://前缀
        file_name = os.path.join(save_directory, object_name.split('/')[-1])  # 生成完整的本地路径
        download_file(bucket_name, object_name, file_name)

except FileNotFoundError:
    print(f"文件未找到: {file_path}")
except Exception as e:
    print(f"发生错误: {e}")
