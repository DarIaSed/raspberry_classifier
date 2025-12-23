import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import os

st.title("🍓 Классификатор видов малины")
st.write("Загрузите фото малины, и модель определит её вид")

@st.cache_resource
def load_model():
    classes = ['rubus_idaeus', 'rubus_occidentalis', 'rubus_parviflorus',
               'rubus_phoenicolasius', 'rubus_spectabilis']
    russian_names = {
        'rubus_idaeus': 'Малина обыкновенная',
        'rubus_occidentalis': 'Малина западная (черная)',
        'rubus_parviflorus': 'Малина мелкоцветковая',
        'rubus_phoenicolasius': 'Малина пурпурноплодная',
        'rubus_spectabilis': 'Малина великолепная'
    }

    class RaspberryClassifierResNet(nn.Module):
        def __init__(self, num_classes):
            super(RaspberryClassifierResNet, self).__init__()
            self.backbone = models.resnet50(weights='IMAGENET1K_V1')
            
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
            
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, num_classes)
            )
        
        def forward(self, x):
            return self.backbone(x)

    model = RaspberryClassifierResNet(num_classes=len(classes))

    model_path = 'raspberry_model.pth'
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        st.success("Модель успешно загружена!")

    model.eval()
    return model, classes, russian_names

def predict(image, model, classes, russian_names):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    results = []
    for i in range(3):
        class_name = classes[top3_idx[i]]
        results.append({
            'вид': russian_names.get(class_name, class_name),
            'точность': float(top3_prob[i] * 100)
        })
    
    return results

uploaded_file = st.file_uploader("Выберите фото малины", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Загруженное фото', use_column_width=True)
    model_data = load_model()
    
    if model_data[0] is not None:
        model, classes, russian_names = model_data
        
        if st.button("Определить вид"):
            predictions = predict(image, model, classes, russian_names)
            
            st.success("Результат:")
            for i, pred in enumerate(predictions):
                st.write(f"{i+1}. **{pred['вид']}** - {pred['точность']:.1f}%")
                st.progress(pred['точность'] / 100)