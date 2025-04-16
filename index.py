from flask import Flask, request , send_from_directory , jsonify, url_for
from deepface import DeepFace
import os
import tempfile

# ### main code
app = Flask(__name__)

#  ### this is for making folder
folder_name = os.path.join('profile_images')
os.makedirs(folder_name, exist_ok=True)


# routes for image upload
@app.route('/upload',methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return {"status": 400,'message': 'Image parameter required'}

    file = request.files['image']
    file.save(os.path.join(folder_name, file.filename))
    return {"status": 200,'message': 'Image save successfully' }



# get image
@app.route('/getimage', methods=['GET'])
def getimage():
    image = request.args.get('image')
    if not image:
        return {"status": 400, "message": "Image parameter required"}
    
    # Generate the full URL for the image
    image_url = request.host_url.rstrip('/') + url_for('serve_image', filename=image)
    return {"status": 200, "url": image_url}

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(folder_name, filename)





# deepface face similarity 
@app.route('/deepface', methods=['POST'])
def verifyFace():
    if 'image1' not in request.files and 'image2' not in request.files:
        return {"status": 400, 'message': 'Both image1 and image2 are required'}
    
    image1 = request.files['image1']
    image2 = request.files['image2']

    # Create temporary files
    temp_dir = tempfile.gettempdir()
    temp_path1 = os.path.join(temp_dir, "temp_img1.jpg")
    temp_path2 = os.path.join(temp_dir, "temp_img2.jpg")

    try:
        # Save files temporarily
        image1.save(temp_path1)
        image2.save(temp_path2)
        
        # Verify faces using file paths
        result = DeepFace.verify(
            img1_path=temp_path1,
            img2_path=temp_path2,
            model_name="VGG-Face",  # Using VGG-Face model for better accuracy
            distance_metric="cosine",  # Using cosine distance metric
            enforce_detection=False
        )
        
        # Get the distance and threshold from the result
        distance = result.get('distance', 1.0)  # Default to 1.0 if not found
        threshold = result.get('threshold', 0.4)  # Default threshold
        
        # Custom verification with stricter threshold
        is_same_person = distance <= threshold
        
        response = {
            "status": 200,
            "verified": is_same_person,
            "distance": distance,
            "threshold": threshold,
            "model": "VGG-Face",
            "similarity_score": ((1 - distance) * 100) if distance <= 1 else 0
        }
        return jsonify(result)
    
    except Exception as e:
        return {"status": 500, "message": str(e)}
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path1):
            os.remove(temp_path1)
        if os.path.exists(temp_path2):
            os.remove(temp_path2)











# @app.route('/deepface', methods=['POST'])
# def deepface():
#     data = request.json
#     image1 = data['image1']
#     image2 = data['image2']

#     result = DeepFace.verify(image1, image2, enforce_detection=False)
#     return jsonify(result)






# Create documents folder
docs_folder = os.path.join('documents')
os.makedirs(docs_folder, exist_ok=True)


# Route for document upload
@app.route('/upload-document', methods=['POST'])
def upload_document():
    if 'document' not in request.files:
        return {"status": 400, "message": "Document file required"}

    file = request.files['document']

    
    # Verify file extension (optional)
    allowed_extensions = {'.pdf', '.doc', '.docx', '.txt'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return {"status": 400, "message": "Invalid file type"}

    file.save(os.path.join(docs_folder, file.filename))
    return {"status": 200, "message": "Document saved successfully"}




# Route to get all documents
@app.route('/get-documents', methods=['GET'])
def get_documents():
    try:
        # Get list of all files in docs_folder
        documents = os.listdir(docs_folder)
        
        # Create URLs for all documents
        document_urls = []
        for doc in documents:
            doc_url = request.host_url.rstrip('/') + url_for('serve_document', filename=doc)
            document_urls.append({
                "filename": doc,
                "url": doc_url
            })
        
        return {"status": 200, "documents": document_urls}
    except Exception as e:
        return {"status": 500, "message": str(e)}

@app.route('/documents/<path:filename>')
def serve_document(filename):
    return send_from_directory(docs_folder, filename)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000,debug=True)