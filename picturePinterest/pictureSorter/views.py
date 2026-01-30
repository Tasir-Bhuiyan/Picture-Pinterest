from django.shortcuts import render, redirect
from .models import MyImageModel
from torchvision import models, transforms
from PIL import Image
import torch
from sklearn.metrics.pairwise import cosine_similarity
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import hashlib
from django.conf import settings
from django.shortcuts import render, redirect
from .models import MyImageModel
from torchvision import models, transforms
from PIL import Image
import torch

threshold = settings.IMAGE_SIMILARITY_THRESHOLD
from pillow_heif import register_heif_opener
register_heif_opener()
# -----------------------------
# Load pre-trained feature extractor
# -----------------------------
feature_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_model = torch.nn.Sequential(*list(feature_model.children())[:-1])  # remove classifier
feature_model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Helper functions
# -----------------------------
def get_file_hash(file):
    """Calculate MD5 hash of uploaded file"""
    hasher = hashlib.md5()
    for chunk in file.chunks():
        hasher.update(chunk)
    return hasher.hexdigest()


def extract_features(image_path):
    try:
        img = Image.open(image_path)
        
        # Ensure image is in RGB (crucial for RAW and HEIC with alpha channels)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img_t = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            feat = feature_model(img_t).squeeze().numpy()
        return feat
    except Exception as e:
        print(f"Error: {e}")
        return None


def group_similar_images():
    """Group images based on cosine similarity"""
    images = list(MyImageModel.objects.all())
    if not images:
        return []

    # Extract features and filter out failed extractions
    features_data = []
    valid_images = []
    for img in images:
        feat = extract_features(img.image.path)
        if feat is not None:
            features_data.append(feat)
            valid_images.append(img)
    
    if not valid_images:
        return []
    
    sim_matrix = cosine_similarity(features_data)

    groups = []
    visited = set()

    for i, img in enumerate(valid_images):
        if i in visited:
            continue
        group = [img]
        visited.add(i)
        for j in range(i+1, len(valid_images)):
            if j not in visited and sim_matrix[i][j] > threshold:
                group.append(valid_images[j])
                visited.add(j)
        groups.append(group)

    print("DEBUG: Groups formed:")
    for g in groups:
        print([img.id for img in g])

    return groups


# -----------------------------
# Views
# -----------------------------
def index(request):
    """Upload page with duplicate detection"""
    if request.method == "POST":
        images = request.FILES.getlist("images")
        
        # Optional: Limit uploads
        if len(images) > 50:
            return render(request, "upload.html", {
                'error': 'Maximum 50 images allowed per upload. Please select fewer images.'
            })
        
        # Upload images with duplicate detection
        for img in images:
            file_hash = get_file_hash(img)
            # Check if hash exists in database to avoid duplicates
            if not MyImageModel.objects.filter(file_hash=file_hash).exists():
                MyImageModel.objects.create(image=img, file_hash=file_hash)
        
        return redirect("compare")
    
    # GET request - show upload form
    return render(request, "upload.html")


def compare(request):
    """Compare page: show pairs to choose left/right"""
    # Check if there are any images uploaded
    if not MyImageModel.objects.exists():
        return redirect('upload')
    
    # Initialize session if first time or session expired
    if 'groups' not in request.session:
        groups = group_similar_images()
        if not groups:
            return redirect('upload')
        # Store groups as lists of image IDs
        request.session['groups'] = [[img.id for img in group] for group in groups]
        request.session['survivors'] = request.session['groups'][:]

    survivors = request.session['survivors']

    # Find first group with 2+ images remaining
    pair = None
    group_index = None
    for idx, group in enumerate(survivors):
        if len(group) >= 2:
            # Always take first two images in the group
            pair = group[0:2]
            group_index = idx
            break

    # No more pairs? Go to results
    if pair is None:
        return redirect('results')

    # Get image URLs
    try:
        img1 = MyImageModel.objects.get(id=pair[0]).image.url
        img2 = MyImageModel.objects.get(id=pair[1]).image.url
    except MyImageModel.DoesNotExist:
        # Images were deleted, reset session
        request.session.flush()
        return redirect('upload')

    return render(request, 'compare.html', {
        'img1': img1,
        'img2': img2,
        'group_index': group_index
    })


@csrf_exempt
def select_image(request):
    """Handle image selection in tournament"""
    if request.method == 'POST':
        data = request.POST
        group_index = int(data['group_index'])
        choice = int(data['choice'])  # 0 = left won, 1 = right won

        survivors = request.session.get('survivors', [])
        
        if group_index >= len(survivors):
            return JsonResponse({'status': 'error'}, status=400)
        
        group = survivors[group_index]
        
        if len(group) < 2:
            return JsonResponse({'status': 'error'}, status=400)
        
        # Get first two image IDs (always comparing first two)
        left_id = group[0]
        right_id = group[1]
        
        # Determine loser and remove from group
        loser_id = right_id if choice == 0 else left_id
        group.remove(loser_id)
        
        # Update survivors
        survivors[group_index] = group
        request.session['survivors'] = survivors
        request.session.modified = True
        
        return JsonResponse({'status': 'ok'})
    
    return JsonResponse({'status': 'error'}, status=400)


def results(request):
    """Show final best images"""
    survivors = request.session.get('survivors', [])
    final_images = []

    for group in survivors:
        for img_id in group:
            try:
                img = MyImageModel.objects.get(id=img_id)
                final_images.append(img.image.url)
            except MyImageModel.DoesNotExist:
                continue

    return render(request, "results.html", {'images': final_images})


def reset(request):
    """Clear session data and delete all images"""
    request.session.flush()
    MyImageModel.objects.all().delete()
    return redirect('upload')