"""
Cloud storage manager for MRI images and reports
Supports both Firebase Storage and AWS S3
"""

import os
import hashlib
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class CloudStorageManager:
    """Manages cloud storage operations for medical images and reports"""
    
    def __init__(self, storage_type='firebase'):
        """
        Initialize cloud storage manager
        
        Args:
            storage_type (str): 'firebase' or 's3'
        """
        self.storage_type = storage_type
        self.initialized = False
        
        try:
            if storage_type == 'firebase':
                self._init_firebase()
            elif storage_type == 's3':
                self._init_s3()
            else:
                raise Exception(f"Unsupported storage type: {storage_type}")
        except Exception as e:
            logger.warning(f"Cloud storage initialization failed: {str(e)}")
            logger.info("Running in mock mode - files will be stored locally")
    
    def _init_firebase(self):
        """Initialize Firebase Admin SDK"""
        try:
            import firebase_admin
            from firebase_admin import credentials, storage
            
            # Initialize Firebase (you'll need to add your service account key)
            if not firebase_admin._apps:
                # For demo purposes - you'll need to replace with your service account
                cred = credentials.Certificate("path/to/serviceAccountKey.json")
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'your-project.appspot.com'
                })
            
            self.bucket = storage.bucket()
            self.initialized = True
            logger.info("✅ Firebase Storage initialized")
            
        except Exception as e:
            logger.error(f"Firebase initialization failed: {str(e)}")
            raise e
    
    def _init_s3(self):
        """Initialize AWS S3"""
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError
            
            # Initialize S3 client
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            self.bucket_name = os.getenv('AWS_S3_BUCKET', 'alzheimer-prediction-images')
            self.initialized = True
            logger.info("✅ AWS S3 initialized")
            
        except Exception as e:
            logger.error(f"S3 initialization failed: {str(e)}")
            raise e
    
    def store_image(self, image_bytes, image_hash):
        """
        Store MRI image in cloud storage
        
        Args:
            image_bytes (bytes): Image data
            image_hash (str): SHA256 hash of the image
            
        Returns:
            str: URL to stored image
        """
        try:
            if not self.initialized:
                return self._store_locally(image_bytes, image_hash, 'images')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"mri_images/{timestamp}_{image_hash[:12]}.png"
            
            if self.storage_type == 'firebase':
                return self._store_firebase(image_bytes, filename)
            elif self.storage_type == 's3':
                return self._store_s3(image_bytes, filename)
            
        except Exception as e:
            logger.error(f"Image storage failed: {str(e)}")
            return self._store_locally(image_bytes, image_hash, 'images')
    
    def store_report(self, report_data, report_id):
        """
        Store medical report in cloud storage
        
        Args:
            report_data (dict): Report data
            report_id (str): Unique report identifier
            
        Returns:
            str: URL to stored report
        """
        try:
            report_json = json.dumps(report_data, indent=2)
            report_bytes = report_json.encode('utf-8')
            
            if not self.initialized:
                return self._store_locally(report_bytes, report_id, 'reports')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"reports/{timestamp}_{report_id}.json"
            
            if self.storage_type == 'firebase':
                return self._store_firebase(report_bytes, filename)
            elif self.storage_type == 's3':
                return self._store_s3(report_bytes, filename)
            
        except Exception as e:
            logger.error(f"Report storage failed: {str(e)}")
            return self._store_locally(report_bytes, report_id, 'reports')
    
    def _store_firebase(self, data, filename):
        """Store data in Firebase Storage"""
        try:
            blob = self.bucket.blob(filename)
            blob.upload_from_string(data)
            blob.make_public()
            
            url = blob.public_url
            logger.info(f"✅ Stored in Firebase: {filename}")
            return url
            
        except Exception as e:
            logger.error(f"Firebase storage failed: {str(e)}")
            raise e
    
    def _store_s3(self, data, filename):
        """Store data in AWS S3"""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=data,
                ACL='public-read'
            )
            
            url = f"https://{self.bucket_name}.s3.amazonaws.com/{filename}"
            logger.info(f"✅ Stored in S3: {filename}")
            return url
            
        except Exception as e:
            logger.error(f"S3 storage failed: {str(e)}")
            raise e
    
    def _store_locally(self, data, identifier, folder):
        """Store data locally as fallback"""
        try:
            # Create local storage directory
            storage_dir = f"local_storage/{folder}"
            os.makedirs(storage_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            extension = '.png' if folder == 'images' else '.json'
            filename = f"{timestamp}_{identifier[:12]}{extension}"
            filepath = os.path.join(storage_dir, filename)
            
            # Write data
            mode = 'wb' if folder == 'images' else 'w'
            with open(filepath, mode) as f:
                if folder == 'images':
                    f.write(data)
                else:
                    f.write(data.decode('utf-8'))
            
            url = f"file://{os.path.abspath(filepath)}"
            logger.info(f"📁 Stored locally: {filepath}")
            return url
            
        except Exception as e:
            logger.error(f"Local storage failed: {str(e)}")
            return None
    
    def get_image_url(self, image_hash):
        """Retrieve image URL by hash"""
        # Implementation depends on your storage structure
        # This is a placeholder - you'd implement actual lookup logic
        return f"storage_url_for_{image_hash}"
    
    def delete_image(self, filename):
        """Delete image from cloud storage"""
        try:
            if not self.initialized:
                # Delete local file
                filepath = f"local_storage/images/{filename}"
                if os.path.exists(filepath):
                    os.remove(filepath)
                    logger.info(f"🗑️ Deleted local file: {filepath}")
                return True
            
            if self.storage_type == 'firebase':
                blob = self.bucket.blob(filename)
                blob.delete()
            elif self.storage_type == 's3':
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=filename)
            
            logger.info(f"🗑️ Deleted from cloud: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {filename}: {str(e)}")
            return False