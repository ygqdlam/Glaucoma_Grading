
import os
from collections import Counter

def calculate_dice(file1_path, file2_path):
    """计算两个文件的Dice系数"""
    try:
        with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
            content1 = f1.read()
            content2 = f2.read()
            
        # 将文件内容转为字符集合
        set1 = set(content1)
        set2 = set(content2)
        
        # 计算交集和并集大小
        intersection = len(set1 & set2)
        total_elements = len(set1) + len(set2)
        
        # Dice系数公式：2*|A∩B|/(|A|+|B|)
        return 2 * intersection / total_elements if total_elements > 0 else 0.0
        
    except Exception as e:
        print(f"Error processing {file1_path} and {file2_path}: {str(e)}")
        return 0.0

def compare_folders(folder1, folder2):
    """比较两个文件夹中所有文件的Dice系数"""
    results = {}
    
    # 获取两个文件夹的文件列表
    files1 = {f: os.path.join(folder1, f) for f in os.listdir(folder1) 
              if os.path.isfile(os.path.join(folder1, f))}
    files2 = {f: os.path.join(folder2, f) for f in os.listdir(folder2) 
              if os.path.isfile(os.path.join(folder2, f))}
    
    # 找出共同文件名
    common_files = set(files1.keys()) & set(files2.keys())
    
    # 计算每个共同文件的Dice系数
    for filename in common_files:
        dice = calculate_dice(files1[filename], files2[filename])
        results[filename] = dice
    
    return results

if __name__ == "__main__":
    

    folder1='/home/yanggq/project/grading/GlaucomaRecognition-main/stream-gamma/pages/funds/result/Pred'
    folder2='/home/yanggq/project/grading/task3_disc_cup_segmentation/training/Disc_Cup_Mask'


    results = compare_folders(folder1, folder2)
    
    print("\n文件相似度报告(Dice系数):")
    print("{:<30} {:<10}".format('文件名', '相似度'))
    print("-" * 40)
    for filename, dice in results.items():
        print("{:<30} {:.4f}".format(filename, dice))
    
    avg_dice = sum(results.values())/len(results) if results else 0
    print("\n平均相似度: {:.4f}".format(avg_dice))
