import json
from ..utils.api_utils import translate_text_with_ollama

def export_text_draw_result(text_jsons):
    text_draw_results = []
    for item in text_jsons:
        text_contnt = {}
        bbox = item["bbox"]
        text_contnt["x1"] = bbox[0]/10
        text_contnt["y1"] = bbox[1]/10
        text_contnt["x2"] = bbox[2]/10
        text_contnt["y2"] = bbox[3]/10
        text_contnt["page"] = item["page_idx"]+1
        content = ""
        if "text" in item:
            content = item["text"]
        elif "list_items" in item:
            content = ','.join(item["list_items"])
        else:
            content = ""
        text_contnt['content'] = translate_text_with_ollama(content)
        text_draw_results.append(text_contnt)
    return text_draw_results

def export_image_draw_result(image_jsons):
    image_draw_results = []
    for item in image_jsons:
        iamge_contnt = {}
        bbox = item["bbox"]
        iamge_contnt["x1"] = bbox[0]/10
        iamge_contnt["y1"] = bbox[1]/10
        iamge_contnt["x2"] = bbox[2]/10
        iamge_contnt["y2"] = bbox[3]/10
        iamge_contnt["page"] = item["page_idx"]+1
        iamge_contnt['content'] = ""
        image_draw_results.append(iamge_contnt)
    return image_draw_results

def export_reaction_draw_result(image_jsons):
    reaction_draw_results = []
    reactions = image_jsons["reactions"]
    for item in reactions:
        reacion_content = {}
        evidence = item["evidence"]
        bbox = evidence["boxes"][0]
        reacion_content["x1"] = bbox[0]/10
        reacion_content["y1"] = bbox[1]/10
        reacion_content["x2"] = bbox[2]/10
        reacion_content["y2"] = bbox[3]/10
        reacion_content["page"] = evidence["page"]+1
        reacion_content["content"] = f"rxn_smiles: {item['rxn_smiles']}\nexperiments: {item['experiments']}\nsolvent: {item['solvent']}"
        reaction_draw_results.append(reacion_content)
    return reaction_draw_results

def export_compound_draw_rsult(compound_jsons):
    compound_draw_rsults = []
    for item in compound_jsons:
        detect = item["detect"]
        image_bbox = item["bbox"]
        image_bbox_w = image_bbox[2] - image_bbox[0]
        image_bbox_h = image_bbox[3] - image_bbox[1]

        for compound in detect:
            compound_content = {}
            bbox = compound["bbox"]
            width, height = compound["image_resolution"]
            x_scale_0 = bbox[0] / width
            x_scale_1 = bbox[2] / width
            y_scale_0  = bbox[1] / height
            y_scale_1 = bbox[3] / height

            compound_content["x1"] = (image_bbox[0] + image_bbox_w*(x_scale_0))/10
            compound_content["y1"] = (image_bbox[1] + image_bbox_h*(y_scale_0))/10
            compound_content["x2"] = (image_bbox[0] + image_bbox_w*(x_scale_1))/10
            compound_content["y2"] = (image_bbox[1] + image_bbox_h*(y_scale_1))/10
            compound_content["page"] = item["page_idx"]+1
            compound_content["content"] = f"name: {compound['name']}\nsmiles: {compound['smiles']['smiles']}\nconf: {compound['conf']}"
            compound_draw_rsults.append(compound_content)
    
    return compound_draw_rsults

def export_yolo_compound_draw_rsult(compound_jsons):
    compound_draw_rsults = []
    for item in compound_jsons:
        detect = item["detect"]
        image_bbox = item["bbox"]
        image_bbox_w = image_bbox[2] - image_bbox[0]
        image_bbox_h = image_bbox[3] - image_bbox[1]

        for compound in detect:
            # if compound['class_id'] != 0:
            #     continue
            compound_content = {}
            bbox = compound["bbox"]
            width, height = compound["image_resolution"]
            x_scale_0 = bbox[0] / width
            x_scale_1 = bbox[2] / width
            y_scale_0  = bbox[1] / height
            y_scale_1 = bbox[3] / height

            compound_content["x1"] = (image_bbox[0] + image_bbox_w*(x_scale_0))/10
            compound_content["y1"] = (image_bbox[1] + image_bbox_h*(y_scale_0))/10
            compound_content["x2"] = (image_bbox[0] + image_bbox_w*(x_scale_1))/10
            compound_content["y2"] = (image_bbox[1] + image_bbox_h*(y_scale_1))/10
            compound_content["page"] = item["page_idx"]+1
            if compound['class_id'] == 0:
                compound_content["name"] = compound.get("name", "")
                compound_content["smiles"] = compound.get("smiles",{}).get("smiles", "")
                compound_content["confidence"] = compound.get("confidence", 0.0)
            else:
                compound_content["content"] = compound.get("name", "")
            compound_draw_rsults.append(compound_content)
    
    return compound_draw_rsults

def export_extported_result_draw_result(extported_jsons):
    reaction_draw_results = []
    for item in extported_jsons:
        reacion_content = {}
        for key in item:
            if key not in ["bbox", "page"]:
                reacion_content[key] = item[key]
        bbox = item["bbox"]
        reacion_content["x1"] = bbox[0]/10
        reacion_content["y1"] = bbox[1]/10
        reacion_content["x2"] = bbox[2]/10
        reacion_content["y2"] = bbox[3]/10
        reacion_content["page"] = item["page"]+1
        # reacion_content["content"] = f"rxn_smiles: {item["rxn_smiles"]}\nexperiments: {item["experiments"]}\nsolvent: {item["solvent"]}"
        reaction_draw_results.append(reacion_content)
    return reaction_draw_results

# with open("/mnt/binghao/output/2025-11-20_14-58-09_results.json", "r") as f:
#     content = json.loads(f.read())

# data = []

# text_draw_results = export_text_draw_result(content["text_jsons"])
# # reaction_draw_results = export_reaction_draw_result(content["fusion_result"])
# # compound_draw_rsults = export_compound_draw_rsult(content["compound_detections"])
# yolo_draw_results = export_yolo_compound_draw_rsult(content["compound_name_extractions"])
# data = text_draw_results + yolo_draw_results
# print(yolo_draw_results)
# with open("data.json", 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)