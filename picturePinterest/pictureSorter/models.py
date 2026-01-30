from django.db import models

class MyImageModel(models.Model):
    image = models.ImageField(upload_to='uploads/')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Hash for duplicate detection
    file_hash = models.CharField(max_length=32, null=True, blank=True)

    # AI grouping
    group_id = models.IntegerField(null=True, blank=True)

    # tournament tracking
    eliminated = models.BooleanField(default=False)

    def __str__(self):
        return f"Image {self.id}"

