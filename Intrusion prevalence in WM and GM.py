import os
import cv2
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import random

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class NPPSegmentationDataset:
    def __init__(self, root_dir, wavelengths=['550', '600'], tissue_types=["TU", "Irregular"],
                 gmwm_mapping_path=None, tcc_mapping_path=None):
        """
        Initialize the NPP Segmentation Dataset.
        
        Parameters:
            root_dir (str): The root directory of the NPP dataset.
            wavelengths (list): A list of wavelength folder names (e.g., ['550', '600']).
            tissue_types (list): List of tissue types to include (e.g., ["TU", "Irregular"]).
                                HT (healthy tissue) samples are excluded.
            gmwm_mapping_path (str): Path to the GM_WM JSON mapping file.
            tcc_mapping_path (str): Path to the TCC JSON mapping file.
        """
        self.root_dir = root_dir
        self.wavelengths = wavelengths
        self.tissue_types = tissue_types
        self.samples = {}
        self.load_samples()
        
        if gmwm_mapping_path is None or tcc_mapping_path is None:
            raise ValueError("Both GM_WM and TCC mapping file paths must be provided.")
        self.gmwm_mapping = self.load_color_mapping(gmwm_mapping_path)
        self.tcc_mapping = self.load_color_mapping(tcc_mapping_path)

    @staticmethod
    def load_color_mapping(mapping_file_path):
        """
        Load a color mapping from a JSON file.
        
        Args:
            mapping_file_path (str): Path to the JSON file.
            
        Returns:
            dict: Mapping from label names to RGB color values.
        """
        with open(mapping_file_path, 'r') as f:
            mapping = json.load(f)
        return mapping

    def load_samples(self):
        """
        Traverse the dataset folder structure and load samples from each specified wavelength
        and tissue type. The samples are stored in a dictionary with composite keys:
            <sample_id>_<wavelength>
        """
        self.samples = {}
        for wavelength in self.wavelengths:
            wavelength_folder = os.path.join(self.root_dir, wavelength)
            if not os.path.isdir(wavelength_folder):
                logging.warning(f"Wavelength folder not found: {wavelength_folder}. Skipping '{wavelength}'.")
                continue

            for tissue in self.tissue_types:
                tissue_folder = os.path.join(wavelength_folder, tissue)
                if not os.path.isdir(tissue_folder):
                    logging.warning(f"Tissue folder not found: {tissue_folder}. Skipping '{tissue}'.")
                    continue

                for sample_name in os.listdir(tissue_folder):
                    sample_path = os.path.join(tissue_folder, sample_name)
                    if not os.path.isdir(sample_path):
                        continue
                    gmwm_path = os.path.join(sample_path, 'histology', 'GM_WM.png')
                    tcc_path = os.path.join(sample_path, 'histology', 'TCC.png')
                    if not os.path.exists(gmwm_path):
                        logging.warning(f"GM_WM file missing for sample {sample_name} (wavelength {wavelength}).")
                        continue
                    if not os.path.exists(tcc_path):
                        logging.warning(f"TCC file missing for sample {sample_name} (wavelength {wavelength}).")
                        continue
                    composite_key = f"{sample_name}_{wavelength}"
                    self.samples[composite_key] = {
                        'sample_id': sample_name,
                        'tissue_type': tissue,
                        'wavelength': wavelength,
                        'gmwm_path': gmwm_path,
                        'tcc_path': tcc_path
                    }
        logging.info(f"Loaded {len(self.samples)} samples across wavelengths: {self.wavelengths}.")

    def plot_sample_images(self, composite_sample_id):
        """
        Display the GM_WM and TCC images side by side for a specified sample.
        """
        if composite_sample_id not in self.samples:
            raise KeyError(f"Sample key '{composite_sample_id}' not found.")
        sample = self.samples[composite_sample_id]
        gmwm_img = cv2.imread(sample['gmwm_path'], cv2.IMREAD_COLOR)
        tcc_img = cv2.imread(sample['tcc_path'], cv2.IMREAD_COLOR)
        if gmwm_img is None:
            raise FileNotFoundError(f"GM_WM image not found: {sample['gmwm_path']}")
        if tcc_img is None:
            raise FileNotFoundError(f"TCC image not found: {sample['tcc_path']}")
        gmwm_img = cv2.cvtColor(gmwm_img, cv2.COLOR_BGR2RGB)
        tcc_img = cv2.cvtColor(tcc_img, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(gmwm_img)
        axes[0].set_title(f"{composite_sample_id}\nGM_WM")
        axes[0].axis('off')
        axes[1].imshow(tcc_img)
        axes[1].set_title(f"{composite_sample_id}\nTCC")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

    def compute_iz_prevalence(self, composite_sample_id, debug=False):
        """
        Compute the percentage of IZ pixels in GM and WM regions for a specified sample.
        If debug is True, display two rows of diagnostic images:
          - Row 1: Original GM_WM image, GM mask, WM mask.
          - Row 2: Original TCC image, IZ mask.
          
        Args:
            composite_sample_id (str): Composite sample key (<sample_id>_<wavelength>).
            debug (bool): Whether to display diagnostic plots (default False).
        
        Returns:
            dict: A dictionary with IZ prevalence percentages for GM and WM.
        """
        if composite_sample_id not in self.samples:
            raise KeyError(f"Sample key '{composite_sample_id}' not found.")
        sample = self.samples[composite_sample_id]
        
        # Load and convert segmentation images to RGB.
        gmwm_img = cv2.imread(sample['gmwm_path'], cv2.IMREAD_COLOR)
        if gmwm_img is None:
            raise FileNotFoundError(f"GM_WM image not found: {sample['gmwm_path']}")
        gmwm_img = cv2.cvtColor(gmwm_img, cv2.COLOR_BGR2RGB)
        
        tcc_img = cv2.imread(sample['tcc_path'], cv2.IMREAD_COLOR)
        if tcc_img is None:
            raise FileNotFoundError(f"TCC image not found: {sample['tcc_path']}")
        tcc_img = cv2.cvtColor(tcc_img, cv2.COLOR_BGR2RGB)
        
        # Retrieve expected RGB colors from the mappings.
        gm_color = np.array(self.gmwm_mapping["GM"])
        wm_color = np.array(self.gmwm_mapping["WM"])
        iz_color = np.array(self.tcc_mapping["IZ"])
        
        # Create boolean masks by comparing each pixel's color.
        gm_mask = np.all(gmwm_img == gm_color, axis=-1)
        wm_mask = np.all(gmwm_img == wm_color, axis=-1)
        iz_mask = np.all(tcc_img == iz_color, axis=-1)
        
        # If debugging, display diagnostic plots.
        if debug:
            fig = plt.figure(figsize=(18, 10))
            gs = gridspec.GridSpec(2, 3)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(gmwm_img)
            ax1.set_title("Original GM_WM Image")
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(gm_mask, cmap='gray')
            ax2.set_title("GM Mask")
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(wm_mask, cmap='gray')
            ax3.set_title("WM Mask")
            ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(tcc_img)
            ax4.set_title("Original TCC Image")
            ax4.axis('off')
            
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(iz_mask, cmap='gray')
            ax5.set_title("IZ Mask")
            ax5.axis('off')
            
            # Leave the last cell empty.
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Compute intersection masks and percentages.
        gm_iz_mask = np.logical_and(gm_mask, iz_mask)
        wm_iz_mask = np.logical_and(wm_mask, iz_mask)
        gm_total = np.sum(gm_mask)
        wm_total = np.sum(wm_mask)
        gm_iz_percentage = (np.sum(gm_iz_mask) / gm_total * 100) if gm_total else 0
        wm_iz_percentage = (np.sum(wm_iz_mask) / wm_total * 100) if wm_total else 0
        
        return {
            'composite_sample_id': composite_sample_id,
            'gm_iz_percentage': gm_iz_percentage,
            'wm_iz_percentage': wm_iz_percentage
        }
    
    def compute_global_statistics(self, debug=False):
        """
        Compute and aggregate IZ prevalence statistics over all samples. In addition to 
        overall statistics (across all samples), this method computes statistics for each tissue type.
        The final output includes six groups:
        
            Overall GM, Overall WM,
            TU GM, TU WM,
            Irregular GM, Irregular WM.
        
        For each group, the mean and standard deviation of the IZ percentages are computed.
        
        Returns:
            tuple: (DataFrame of per-sample statistics, dictionary of global summary statistics)
        """
        results = []
        for composite_sample_id in self.samples:
            stats = self.compute_iz_prevalence(composite_sample_id, debug=debug)
            # Include tissue type information for grouping.
            stats['tissue_type'] = self.samples[composite_sample_id]['tissue_type']
            results.append(stats)
        
        df = pd.DataFrame(results)
        
        global_stats = {}
        # Overall statistics (across all samples).
        overall_gm_mean = df['gm_iz_percentage'].mean()
        overall_gm_std  = df['gm_iz_percentage'].std()
        overall_wm_mean = df['wm_iz_percentage'].mean()
        overall_wm_std  = df['wm_iz_percentage'].std()
        global_stats['Overall GM'] = {'mean': overall_gm_mean, 'std': overall_gm_std, 'n': len(df)}
        global_stats['Overall WM'] = {'mean': overall_wm_mean, 'std': overall_wm_std, 'n': len(df)}
        
        # Group statistics by tissue type.
        for tissue, group_df in df.groupby('tissue_type'):
            gm_mean = group_df['gm_iz_percentage'].mean()
            gm_std  = group_df['gm_iz_percentage'].std()
            wm_mean = group_df['wm_iz_percentage'].mean()
            wm_std  = group_df['wm_iz_percentage'].std()
            global_stats[f"{tissue} GM"] = {'mean': gm_mean, 'std': gm_std, 'n': len(group_df)}
            global_stats[f"{tissue} WM"] = {'mean': wm_mean, 'std': wm_std, 'n': len(group_df)}
        
        return df, global_stats


if __name__ == '__main__':

    root_dataset_dir = '/Users/gianmarcoalbano/Desktop/Horao/NPP'
    gmwm_mapping_path = os.path.join(root_dataset_dir, 'colors', 'GM_WM.json')
    tcc_mapping_path = os.path.join(root_dataset_dir, 'colors', 'TCC.json')
    
    dataset = NPPSegmentationDataset(root_dir=root_dataset_dir,
                                      wavelengths=['550', '600'],
                                      gmwm_mapping_path=gmwm_mapping_path,
                                      tcc_mapping_path=tcc_mapping_path)
    
    # Option to pick a random sample and debug its masks.
    if not dataset.samples:
        print("No samples loaded.")
    else:
        random_sample_key = random.choice(list(dataset.samples.keys()))
        print("Randomly selected sample:", random_sample_key)
        dataset.plot_sample_images(random_sample_key)
        iz_stats = dataset.compute_iz_prevalence(random_sample_key, debug=True)
        print(f"\nIZ Prevalence for sample {iz_stats['composite_sample_id']}:")
        print(f"  GM IZ Percentage: {iz_stats['gm_iz_percentage']:.2f}%")
        print(f"  WM IZ Percentage: {iz_stats['wm_iz_percentage']:.2f}%")
    
    # Compute and display global statistics.
    df_stats, global_stats = dataset.compute_global_statistics()
    print("\nGlobal IZ Prevalence Statistics:")
    for group, stats in global_stats.items():
        print(f"{group}: mean={stats['mean']:.2f}%, std={stats['std']:.2f}%, n={stats['n']}")
    
    print("\nPer-sample statistics (first 5 rows):")
    print(df_stats.head())






