"""
REST API for the Discovery Engine.

This provides a simple Flask-based API for the Discovery Engine.
For production, consider using FastAPI or integrating with your existing service.

Usage:
    python -m mosaic.discovery.api --port 8080

Endpoints:
    POST /discover       - Run discovery on provided data
    POST /compare        - Compare segments
    GET  /health         - Health check
"""

import json
import argparse
from typing import Dict, Any
from dataclasses import asdict

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .engine import DiscoveryEngine, DiscoveryConfig
from .models import DiscoveryReport


def create_app(config: DiscoveryConfig = None) -> "Flask":
    """Create Flask application with discovery endpoints."""
    if not FLASK_AVAILABLE:
        raise ImportError("Flask is required for the API. Install with: pip install flask")

    app = Flask(__name__)
    engine = DiscoveryEngine(config or DiscoveryConfig())

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "service": "mosaic-discovery"})

    @app.route("/discover", methods=["POST"])
    def discover():
        """
        Run discovery on provided data.

        Request body:
        {
            "sessions": {
                "session_id": [
                    {"article_id": "abc", "timestamp": 1234567890},
                    ...
                ],
                ...
            },
            "catalog": {
                "article_id": {"category": "sports", ...},
                ...
            },
            "dataset_name": "my_dataset"  // optional
        }

        Response:
        {
            "metadata": {...},
            "summary": {...},
            "discoveries": [...],
            "top_opportunities": [...],
            "top_oversupply": [...]
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "Request body is required"}), 400

            sessions = data.get("sessions")
            catalog = data.get("catalog")
            dataset_name = data.get("dataset_name", "api_request")

            if not sessions:
                return jsonify({"error": "sessions field is required"}), 400
            if not catalog:
                return jsonify({"error": "catalog field is required"}), 400

            report = engine.discover(sessions, catalog, dataset_name)
            return jsonify(report.to_dict())

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/compare", methods=["POST"])
    def compare():
        """
        Compare preference lifts across segments.

        Request body:
        {
            "sessions": {...},
            "catalog": {...}
        }

        Response:
        {
            "category_name": {
                "morning": 1.5,
                "evening": 0.8,
                "all": 1.2
            },
            ...
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "Request body is required"}), 400

            sessions = data.get("sessions")
            catalog = data.get("catalog")

            if not sessions:
                return jsonify({"error": "sessions field is required"}), 400
            if not catalog:
                return jsonify({"error": "catalog field is required"}), 400

            comparison = engine.compare_segments(sessions, catalog)
            return jsonify(comparison)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/opportunities", methods=["POST"])
    def opportunities():
        """
        Get only high-value opportunities (shortcut endpoint).

        Request body: Same as /discover

        Response:
        {
            "opportunities": [
                {
                    "category": "weather",
                    "segment": "morning",
                    "preference_lift": 2.01,
                    "action": "PROMOTE",
                    "insight": "...",
                    "expected_boost": "+101%"
                },
                ...
            ]
        }
        """
        try:
            data = request.get_json()

            if not data:
                return jsonify({"error": "Request body is required"}), 400

            sessions = data.get("sessions")
            catalog = data.get("catalog")
            top_n = data.get("top_n", 10)

            if not sessions:
                return jsonify({"error": "sessions field is required"}), 400
            if not catalog:
                return jsonify({"error": "catalog field is required"}), 400

            report = engine.discover(sessions, catalog, "api_request")
            opportunities = [
                d.to_api_response()
                for d in report.top_opportunities(top_n)
            ]

            return jsonify({"opportunities": opportunities})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def main():
    """Run the API server."""
    parser = argparse.ArgumentParser(description="MOSAIC Discovery Engine API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if not FLASK_AVAILABLE:
        print("ERROR: Flask is required. Install with: pip install flask")
        return

    app = create_app()
    print(f"Starting MOSAIC Discovery Engine API on {args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  POST /discover      - Run full discovery")
    print(f"  POST /compare       - Compare segments")
    print(f"  POST /opportunities - Get top opportunities")
    print(f"  GET  /health        - Health check")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
