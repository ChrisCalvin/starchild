AMAIE API Structures Report

  This report defines the standardized API structures and conventions for the
  Active Multi-scale Adaptive Inference Engine (AMAIE) system. Its purpose is
  to ensure consistency, predictability, and ease of integration across all
  internal components and external services. Adhering to these guidelines will
  streamline development, reduce errors, and improve the overall
  maintainability and scalability of the system.

  ---

  1. Core Principles

   * RESTful Design: APIs should generally follow RESTful principles, using
     standard HTTP methods (GET, POST, PUT, DELETE) for resource manipulation.
   * Clear Resource Naming: Use clear, plural nouns for resource endpoints (e.g.,
     /orders, /agents).
   * Statelessness: Each request from a client to a server must contain all the
     information needed to understand the request.
   * Layered System: Components should interact through well-defined interfaces,
     promoting modularity.
   * Idempotency: PUT and DELETE operations should be idempotent. POST operations
     are generally not.
   * Security by Design: All APIs must consider security from the outset,
     including authentication, authorization, and data encryption.

  ---

  2. API Categories

  AMAIE APIs can be broadly categorized into:

   * Internal APIs: Used for communication between different AMAIE microservices or
      components (e.g., amaie-core to amaie-trader, launcher to amaie-trader).
     These might leverage lightweight messaging queues (like MQTT, as indicated in
     the project structure) or direct HTTP/RPC calls.
   * External APIs: Exposed to external systems or user interfaces (e.g., trading
     exchanges, data providers, dashboard frontends). These typically use
     HTTP/HTTPS.

  ---

  3. Common API Elements

  3.1. Naming Conventions

   * Endpoints:
       * Lowercase, plural nouns.
       * Use hyphens for separation (kebab-case).
       * Example: /api/v1/trade/orders, /api/v1/market-data/candlesticks
   * Parameters (Query & Path):
       * Lowercase, snake\_case.
       * Example: symbol=BTC_USDT, order_id=12345
   * Request/Response Body Fields:
       * Lowercase, snake\_case.
       * Align with DATA_STRUCTURES field names where applicable.
       * Example: order_type, execution_price, belief_state_id

  3.2. Request/Response Formats

   * Primary Format: JSON (application/json) for all request and response bodies.
   * Character Encoding: UTF-8.
   * Timestamps: All timestamps should be UTC, ISO 8601 format (e.g.,
     "2025-07-30T14:30:00.123456Z").

  3.3. Authentication & Authorization

   * External APIs:
       * Authentication: OAuth 2.0 (Client Credentials, Authorization Code Flow)
         or API Key-based authentication.
       * Authorization: Role-Based Access Control (RBAC) or Attribute-Based Access
         Control (ABAC) to define permissions.
   * Internal APIs:
       * May rely on network-level security (e.g., VPN, private subnets) and
         mutual TLS (mTLS) for service-to-service authentication.
       * Least privilege principle: Services should only have access to what they
         strictly need.

  3.4. Error Handling

  Standardized error responses are crucial for robust client-side error handling.


   * HTTP Status Codes: Use appropriate HTTP status codes to indicate the general
     nature of the error (e.g., 400 Bad Request, 401 Unauthorized, 403 Forbidden,
     404 Not Found, 500 Internal Server Error, 503 Service Unavailable).
   * Error Response Body: A consistent JSON structure for error details.

    1     {
    2       "error": {
    3         "code": "string",        // A unique, machine-readable error
      code (e.g., "INVALID_PARAMETER", "ORDER_NOT_FOUND")
    4         "message": "string",     // A human-readable message 
      describing the error
    5         "details": {             // Optional: More specific details,
      e.g., validation errors for specific fields
    6           "field_name": "error_description",
    7           "another_field": "another_error"
    8         },
    9         "timestamp": "string"    // UTC ISO 8601 timestamp of when 
      the error occurred
   10       }
   11     }

  3.5. Versioning

   * URI Versioning: Use /api/v{major_version}/ in the URI.
   * Backward Compatibility: Strive for backward compatibility within a major
     version. Breaking changes necessitate a new major version.
   * Minor Versions: Handled internally or through documentation updates, not
     typically in the URI.

  3.6. Data Types

  Map Python/PyTorch types to standard JSON types:

   * str, uuid.UUID: JSON string
   * int, float: JSON number
   * bool: JSON boolean
   * list, tuple: JSON array
   * dict: JSON object
   * datetime: JSON string (ISO 8601 UTC)
   * torch.Tensor: Represented as a JSON array (list of lists for
     multi-dimensional tensors) or base64 encoded string for large tensors,
     depending on performance and use case. For small, fixed-shape tensors (like
     market_state_t1), a JSON array is preferred.

  ---

  4. API Templates/Examples

  These examples illustrate how the core data structures (Observation,
  BeliefState, Action, Policy) would be used within API contexts.

  4.1. Trading API: Place Order (External)

   * Description: Allows external systems (e.g., a trading bot, manual interface)
     to submit a trading action to the amaie-trader's execution engine.
   * Endpoint: /api/v1/trade/orders
   * Method: POST

  Request Body (JSON - based on `Action` data structure):

    1 {
    2   "command": "create_order",
    3   "params": {
    4     "symbol": "BTC/USDT",
    5     "side": "buy",
    6     "amount": 0.001,
    7     "order_type": "limit",
    8     "price": 30000.50,
    9     "client_order_id": "unique_id_from_client_123"
   10   },
   11   "priority": "HIGH",
   12   "timeout": 15.0,
   13   "metadata": {
   14     "source_system": "external_trading_app"
   15   }
   16 }

  Successful Response (HTTP 201 Created - includes `Action` status and result):

    1 {
    2   "action_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
    3   "status": "EXECUTING",
    4   "timestamp": "2025-07-30T14:35:00.123456Z",
    5   "command": "create_order",
    6   "params": {
    7     "symbol": "BTC/USDT",
    8     "side": "buy",
    9     "amount": 0.001,
   10     "order_type": "limit",
   11     "price": 30000.50,
   12     "client_order_id": "unique_id_from_client_123"
   13   },
   14   "execution_result": {
   15     "exchange_order_id": "EXCH_ORDER_XYZ",
   16     "status": "NEW",
   17     "filled_amount": 0.0
   18   }
   19 }

  Error Response (HTTP 400 Bad Request):

    1 {
    2   "error": {
    3     "code": "VALIDATION_ERROR",
    4     "message": "Invalid order parameters provided.",
    5     "details": {
    6       "amount": "Must be greater than 0",
    7       "price": "Price is below minimum tick size"
    8     },
    9     "timestamp": "2025-07-30T14:35:01.789012Z"
   10   }
   11 }

  4.2. Market Data API: Get Candlesticks (External)

   * Description: Retrieves historical candlestick data for a given trading pair
     and interval.
   * Endpoint: /api/v1/market-data/candlesticks
   * Method: GET

  Request (Query Parameters):

  GET /api/v1/market-data/candlesticks?symbol=BTC_USDT&interval=1h&start_time=2
  025-07-29T00:00:00Z&end_time=2025-07-30T00:00:00Z

  Successful Response (HTTP 200 OK):

    1 [
    2   {
    3     "timestamp": "2025-07-29T00:00:00Z",
    4     "open": 29500.0,
    5     "high": 29650.0,
    6     "close": 29600.0,
    7     "low": 29450.0,
    8     "volume": 123.45
    9   },
   10   {
   11     "timestamp": "2025-07-29T01:00:00Z",
   12     "open": 29600.0,
   13     "high": 29700.0,
   14     "close": 29680.0,
   15     "low": 29580.0,
   16     "volume": 98.76
   17   }
   18   // ... more candlestick data
   19 ]

  4.3. Agent Control API: Update Agent Configuration (Internal)

   * Description: Allows the launcher or a management interface to update the
     amaie-core agent's runtime configuration (e.g., model parameters, planner
     settings).
   * Endpoint: /api/v1/agent/config
   * Method: PUT

  Request Body (JSON - partial config update):

    1 {
    2   "model_config": {
    3     "damping_factor": 0.6,
    4     "max_iterations": 200
    5   },
    6   "planner_config": {
    7     "cem": {
    8       "num_iterations": 15
    9     }
   10   }
   11 }

  Successful Response (HTTP 200 OK):

   1 {
   2   "status": "success",
   3   "message": "Agent configuration updated successfully.",
   4   "updated_fields": ["model_config.damping_factor",
     "model_config.max_iterations", "planner_config.cem.num_iterations"]
   5 }

  4.4. Health/Status API: Get System Status (Internal/External)

   * Description: Provides a comprehensive overview of the AMAIE system's health
     and operational status.
   * Endpoint: /api/v1/system/status
   * Method: GET

  Successful Response (HTTP 200 OK - simplified example):

    1 {
    2   "timestamp": "2025-07-30T14:40:00.987654Z",
    3   "overall_status": "healthy",
    4   "uptime_seconds": 3600.5,
    5   "profile": {
    6     "name": "development_profile",
    7     "trading_mode": "development"
    8   },
    9   "services": {
   10     "mqtt_broker": {
   11       "status": "running",
   12       "health": "healthy",
   13       "uptime": "3600s"
   14     },
   15     "hummingbot_connector": {
   16       "status": "running",
   17       "health": "healthy",
   18       "uptime": "3580s"
   19     },
   20     "amaie_agent": {
   21       "status": "running",
   22       "health": "healthy",
   23       "uptime": "3500s",
   24       "current_belief_state_id": "some_belief_state_uuid",
   25       "last_policy_id": "some_policy_uuid"
   26     }
   27   },
   28   "health_checks_summary": {
   29     "total_checks": 10,
   30     "passed_checks": 10,
   31     "failed_checks": 0,
   32     "warning_checks": 0
   33   },
   34   "safety_system_status": {
   35     "is_enabled": true,
   36     "safety_level": "NORMAL",
   37     "emergency_stop_active": false
   38   }
   39 }

  ---

  This report serves as a foundational document for designing and implementing
  APIs within the AMAIE project. Consistent application of these guidelines
  will lead to a more robust, maintainable, and scalable system.

