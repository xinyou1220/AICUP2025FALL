from totalsegmentator.python_api import totalsegmentator
import multiprocessing as mp

def run_seg():
    inp = r"C:\Users\coffee\test.nii.gz"
    out = r"C:\Users\coffee\testmask.nii.gz"
    totalsegmentator(
        input=inp,
        output=out,
        task="total",          # 全身任務
        roi_subset=["heart"],  # 只取 heart
        fast=False,            # 高品質
        preview=False
    )

if __name__ == "__main__":
    mp.freeze_support()
    run_seg()
