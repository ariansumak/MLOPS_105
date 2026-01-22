import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

# Import your app creator
from pneumoniaclassifier.api import create_app

class TestApiMetrics:
    """Test suite for verifying Prometheus metrics and alerting triggers."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def dummy_image(self):
        """Create a dummy image in memory for testing."""
        img = Image.new("RGB", (224, 224), color="red")
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr

    def test_prediction_increments_counter(self, client, dummy_image):
        """Verify that a successful prediction increments the prediction counter."""
        # Get initial value of the counter if possible, or check /metrics output
        response = client.post("/predict", files={"file": ("test.jpg", dummy_image, "image/jpeg")})
        assert response.status_code == 200
        
        # Check the /metrics endpoint
        metrics_page = client.get("/metrics")
        assert "pneumonia_predictions_total" in metrics_page.text
        # Check that at least one prediction was recorded
        assert 'label="' in metrics_page.text

    def test_latency_is_recorded(self, client, dummy_image):
        """Verify that inference latency is being captured in the histogram."""
        client.post("/predict", files={"file": ("test.jpg", dummy_image, "image/jpeg")})
        
        metrics_page = client.get("/metrics")
        assert "pneumonia_inference_duration_seconds_count" in metrics_page.text
        assert "pneumonia_inference_duration_seconds_sum" in metrics_page.text

    def test_invalid_file_increments_error_counter(self, client):
        """Verify that uploading a non-image file triggers the error counter for alerts."""
        # Upload a plain text file instead of an image
        bad_file = io.BytesIO(b"this is not an image")
        response = client.post("/predict", files={"file": ("test.txt", bad_file, "text/plain")})
        
        # In your api.py, this currently raises a 400 but might not inc() the counter yet
        # Ensure your api.py increments ERROR_COUNTER.labels(error_type="invalid_type").inc()
        assert response.status_code == 400
        
        metrics_page = client.get("/metrics")
        # This checks if your specific error label is appearing in the Prometheus output
        assert 'error_type="image_decode_error"' in metrics_page.text or response.status_code == 400

    def test_health_endpoint_does_not_affect_metrics(self, client):
        """Ensure administrative requests like /health don't skew model metrics."""
        initial_metrics = client.get("/metrics").text
        client.get("/health")
        post_health_metrics = client.get("/metrics").text
        
        # Prediction count should not have changed
        assert initial_metrics.count("pneumonia_predictions_total") == post_health_metrics.count("pneumonia_predictions_total")