import sys
import os
import argparse
import csv
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def get_opt_config1(opt): # 84.71%
    opt.addAtomIndices =False
    opt.addstereoAnnotation = True
    opt.useBWAtomPalette()
    opt.explicitMethyl = False
    opt.bondLinewidth=2
    opt.fixedBondLength=25
    opt.padding = 0.02
    return opt

def get_opt_config2(opt):
    opt.useDefaultAtomPalette()  # 78.97%
    opt.bondLinewidth = 1.5
    opt.fontSize = 0.8
    return opt

def get_opt_config3(opt): # 80.78%
    opt.fixedBondLength = 22
    opt.scaleBondWidth = False
    opt.maxFontSize = 14
    opt.clearBackground = False
    return opt

def get_opt_config4(opt): # 75.25%
    opt.addStereoAnnotation = True
    opt.addAtomIndices = True
    opt.includeAtomTags = True
    opt.explicitHydrogens = True
    return opt

def get_opt_config5(opt):
    opt.useDefaultAtomPalette()
    opt.bondLineWidth = 1.5
    opt.fixedBondLength = 23
    opt.addStereoAnnotation = True
    opt.explicitMethyl = False
    opt.fontSize = 0.8
    opt.minFontSize = 12
    opt.maxFontSize = 18
    opt.clearBackground = True

    return opt



def generate_image(smiles, output_path, width=1600, height=900, fmt="svg"):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            # print(f"Skipping Invalid SMILES: {smiles}")
            return False
        
        # User options
        w, h = width, height
        
        drawer = None
        if fmt == "svg":
            drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        elif fmt == "png":
            try:
                drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
            except AttributeError:
                print("Error: MolDraw2DCairo not available. Skipping PNG.")
                return False
        else:
            return False

        opt = drawer.drawOptions()
        opt = get_opt_config1(opt)
        # opt.addAtomIndices =False
        # opt.addstereoAnnotation = True
        # opt.useBWAtomPalette()
        # opt.explicitMethyl = False
        # opt.bondLinewidth=2
        # opt.fixedBondLength=25
        # opt.padding = 0.02
        
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        # rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol)
        # content = drawer.FinishDrawing()

        content = drawer.GetDrawingText()
        
        mode = "w" if fmt == "svg" else "wb"
        encoding = "utf-8" if fmt == "svg" else None
        
        with open(output_path, mode, encoding=encoding) as f:
            f.write(content)
            
        return True
    except Exception as e:
        print(f"Error generating {smiles}: {e}")
        return False

def process_batch(input_csv, output_dir, width, height, fmt):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    count = 0
    success = 0
    
    smiles_path_list = []
    print(f"Processing batch from {input_csv}...")
    
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            smiles = row[0].strip()
            if not smiles: continue
            
            # Use index as filename to avoid filesystem issues with SMILES strings
            filename = os.path.join(output_dir, f"{count}.{fmt}")
            
            if generate_image(smiles, filename, width, height, fmt):
                success += 1
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} items... (Success: {success})", end='\r')
            
            smiles_path_list.append({"smiles": smiles, "path": filename})
            if count >= 1000:
                print("Reached 10,000 items limit for this batch run.")
                break
    with open(os.path.join(output_dir, "result.json"), 'w', encoding='utf-8', newline='') as f:
        import json
        json.dump(smiles_path_list, f, ensure_ascii=False, indent=2)

    print(f"\nBatch processing complete. {success}/{count} images generated in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMILES to SVG/PNG")
    parser.add_argument("--smiles", help="Single SMILES string")
    parser.add_argument("-i", "--input", help="Input CSV file for batch processing")
    parser.add_argument("-o", "--output", help="Output filename (for single) or Directory (for batch)")
    parser.add_argument("--format", choices=["svg", "png"], default="png", help="Output format (svg or png)")
    parser.add_argument("--width", type=int, default=800, help="Image width")
    parser.add_argument("--height", type=int, default=450, help="Image height")
    
    args = parser.parse_args()
    
    if args.input:
        # Batch mode
        out_dir = args.output if args.output else "output_images"
        process_batch(args.input, out_dir, args.width, args.height, args.format)
    elif args.smiles:
        # Single mode
        out_file = args.output if args.output else f"mol.{args.format}"
        if generate_image(args.smiles, out_file, args.width, args.height, args.format):
            print(f"Saved to {out_file}")
    else:
        print("Error: Provide --smiles or --input")
