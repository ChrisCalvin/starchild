AMAIE Logging & Monitoring Standards

  This document defines the standards for logging and monitoring within the AMAIE system.
  Consistent and effective logging is crucial for debugging, operational visibility,
  performance analysis, and auditing in a production-grade algorithmic trading system.
  Monitoring ensures the system's health, performance, and adherence to critical
  operational parameters.

  ---

  1. Core Principles

   * Actionable Logs: Logs should provide enough context to understand what happened, why
     it happened, and what actions might be needed.
   * Structured Logging: Where possible, use structured logging (e.g., JSON format) to
     facilitate easier parsing, searching, and analysis by automated tools.
   * Appropriate Granularity: Log messages should be at the correct level of detail for
     their intended purpose.
   * Security: Avoid logging sensitive information (e.g., API keys, personal data, full
     order books) at any level.
   * Performance: Logging should have minimal impact on system performance, especially in
     high-frequency components.
   * Centralized Collection: Logs should ideally be collected and aggregated in a
     centralized system for easy access and analysis.

  ---

  2. Logging Levels

  Utilize Python's standard logging module levels appropriately:

   * `DEBUG`: Detailed information, typically of interest only when diagnosing problems. Use
     for fine-grained events, variable values, and internal state changes.
       * Example: "Calculated EFE for policy X: value=0.123, pragmatic=0.05,
         epistemic=0.073"
   * `INFO`: Confirmation that things are working as expected. Use for significant events,
     successful operations, and system state changes.
       * Example: "AMAIE agent started successfully on profile 'live_trading'"
       * Example: "Order placed: symbol=BTC/USDT, side=BUY, amount=0.01"
   * `WARNING`: An indication that something unexpected happened, or indicative of some
     problem in the near future (e.g., 'disk space low'). The software is still working as
     expected.
       * Example: "Market data feed latency exceeding threshold (100ms)"
       * Example: "Risk limit approaching: current_drawdown=4.5%"
   * `ERROR`: Due to a more serious problem, the software has not been able to perform some
     function.
       * Example: "Failed to connect to exchange API: Connection refused"
       * Example: "Model inference failed: Input tensor shape mismatch"
   * `CRITICAL`: A serious error, indicating that the program itself may be unable to
     continue running.
       * Example: "Unrecoverable error in core agent loop, shutting down."
       * Example: "Safety system triggered emergency stop due to max drawdown breach."

  ---

  3. Log Format

  For consistency and ease of parsing, all logs should adhere to a structured format.
  While RichHandler provides good console output, for file and centralized logging, a
  JSON format is preferred.

  3.1. Standard Log Fields

  Every log entry should include, at minimum:

   * timestamp: UTC ISO 8601 format (e.g., "2025-07-30T14:30:00.123456Z").
   * level: The logging level (e.g., "INFO", "ERROR").
   * service: The name of the service or component emitting the log (e.g., "amaie-core",
     "amaie-trader", "launcher").
   * module: The Python module where the log originated (e.g., "perception", "model",
     "execution").
   * message: A concise, human-readable description of the event.
   * trace_id: (Optional, but highly recommended for distributed systems) A unique ID to
     trace a request or operation across multiple services.
   * span_id: (Optional) A unique ID for a specific operation within a trace.

  3.2. Contextual Logging

  Include relevant contextual information as additional key-value pairs in the log
  message. This is crucial for debugging and analysis.

   * Trading Context: symbol, order_id, trade_id, account_id, strategy_id.
   * Agent Context: agent_id, belief_state_id, policy_id.
   * System Context: container_id, profile_name.

  3.3. Example JSON Log Format

    1 {
    2   "timestamp": "2025-07-30T14:30:00.123456Z",
    3   "level": "INFO",
    4   "service": "amaie-trader",
    5   "module": "execution.engine",
    6   "message": "Order placed successfully",
    7   "symbol": "BTC/USDT",
    8   "order_id": "ORD-XYZ-789",
    9   "amount": 0.001,
   10   "price": 30000.50,
   11   "trace_id": "abc123def456"
   12 }

  3.4. Python logging Configuration Example

    1 import logging
    2 import json
    3 from datetime import datetime, timezone
    4 
    5 class JsonFormatter(logging.Formatter):
    6     def format(self, record):
    7         log_record = {
    8             "timestamp": datetime.fromtimestamp(record.created,
      tz=timezone.utc).isoformat(timespec='microseconds'),
    9             "level": record.levelname,
   10             "service": getattr(record, 'service', 'unknown_service'), # 
      Custom attribute
   11             "module": record.name,
   12             "message": record.getMessage(),
   13         }
   14         # Add extra attributes if they exist
   15         for key, value in record.__dict__.items():
   16             if key not in ['name', 'msg', 'levelname', 'levelno', 'pathname',
      'filename',
   17                            'lineno', 'funcName', 'created', 'msecs',
      'relativeCreated',
   18                            'thread', 'threadName', 'processName', 'process',
      'exc_info',
   19                            'exc_text', 'stack_info', 'args', 'module',
      'asctime', 'message'] and not key.startswith('_'):
   20                 log_record[key] = value
   21         return json.dumps(log_record)
   22 
   23 # Configure root logger
   24 logger = logging.getLogger()
   25 logger.setLevel(logging.DEBUG) # Set global minimum level
   26 
   27 # Console handler (for development)
   28 console_handler = logging.StreamHandler()
   29 console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s -
      %(name)s - %(message)s'))
   30 logger.addHandler(console_handler)
   31 
   32 # File handler (for production logs)
   33 file_handler = logging.FileHandler('amaie.log')
   34 file_handler.setFormatter(JsonFormatter())
   35 file_handler.setLevel(logging.INFO) # Only INFO and above to file
   36 logger.addHandler(file_handler)
   37 
   38 # Example usage with custom attributes
   39 logger.info("System initialized", extra={'service': 'launcher',
      'profile_name': 'development'})
   40 logger.error("Failed to process data", extra={'service': 'amaie-core',
      'module': 'perception', 'data_id': 'xyz123'})

  ---

  4. Monitoring

  Monitoring goes beyond logging to provide real-time insights into system health,
  performance, and operational metrics.

  4.1. Key Performance Indicators (KPIs)

   * Trading Performance:
       * Daily/Weekly/Monthly P&L (Profit & Loss)
       * Drawdown (Max, Current)
       * Sharpe Ratio, Sortino Ratio
       * Win Rate, Average Win/Loss
       * Number of Trades, Volume Traded
   * System Health:
       * Service Uptime/Downtime
       * CPU/Memory/Disk Utilization (per container/service)
       * Network Latency (to exchanges, internal services)
       * Error Rates (API calls, internal processing)
       * Queue Sizes (e.g., pending orders, data processing queues)
   * AI/Agent Specific:
       * Model Inference Latency
       * Planner Execution Time
       * Belief State Uncertainty (trend over time)
       * EFE (Expected Free Energy) values (trend over time)
       * Number of Policy Rejections/Retries

  4.2. Monitoring Tools

   * Prometheus/Grafana: For collecting time-series metrics and creating dashboards.
   * ELK Stack (Elasticsearch, Logstash, Kibana): For centralized log aggregation,
     searching, and visualization.
   * Docker Stats: For basic container-level resource monitoring.
   * Custom Dashboards: The launcher's dashboard.py can provide real-time operational
     views.

  4.3. Alerting

  Define clear alerting rules for critical events and thresholds.

   * Critical Alerts: Trigger immediate notifications (e.g., PagerDuty, SMS) for system
     outages, significant P&L deviations, or safety system breaches.
   * Warning Alerts: Trigger notifications (e.g., Slack, email) for performance
     degradation, high error rates, or approaching risk limits.
   * Alert Content: Alerts should be concise, actionable, and link to relevant dashboards
     or logs for further investigation.

  ---

