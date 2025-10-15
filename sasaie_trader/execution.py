import logging
from sasaie_core.api.action import Action, ActionStatus
from hummingbot_api_client import HummingbotAPIClient

logger = logging.getLogger(__name__)

from typing import Dict, Any

class ExecutionEngine:
    """Handles the execution of actions in a domain-specific context."""

    def __init__(self, hummingbot_api: HummingbotAPIClient):
        self.hummingbot_api = hummingbot_api
        self.tracked_orders: Dict[str, str] = {} # Maps action_id to Hummingbot order_id
        logger.info("ExecutionEngine initialized with HummingbotAPIClient client.")

    async def execute_action(self, action: Action) -> Action:
        """
        Executes a given action using the Hummingbot API.
        """
        logger.info(f"Executing action: {action.command} with params: {action.params}")

        try:
            order_id = None
            if action.command == "create_limit_order":
                # Example: create_limit_order expects trading_pair, is_buy, amount, price
                trading_pair = action.params.get("trading_pair")
                is_buy = action.params.get("is_buy")
                amount = action.params.get("amount")
                price = action.params.get("price")

                if not all([trading_pair, amount, price is not None]):
                    raise ValueError("Missing parameters for create_limit_order")

                order_result = await self.hummingbot_api.create_order(
                    trading_pair=trading_pair,
                    is_buy=is_buy,
                    amount=amount,
                    price=price,
                    order_type="LIMIT"
                )
                order_id = order_result.get("order_id")
                action.status = ActionStatus.PENDING # Order is placed, but not necessarily filled
                action.execution_result = {"status": "SUCCESS", "order_result": order_result}
                logger.info(f"Limit order placed: {order_result}")

            elif action.command == "create_market_order":
                # Example: create_market_order expects trading_pair, is_buy, amount
                trading_pair = action.params.get("trading_pair")
                is_buy = action.params.get("is_buy")
                amount = action.params.get("amount")

                if not all([trading_pair, amount]):
                    raise ValueError("Missing parameters for create_market_order")

                order_result = await self.hummingbot_api.create_order(
                    trading_pair=trading_pair,
                    is_buy=is_buy,
                    amount=amount,
                    order_type="MARKET"
                )
                order_id = order_result.get("order_id")
                action.status = ActionStatus.PENDING # Order is placed, but not necessarily filled
                action.execution_result = {"status": "SUCCESS", "order_result": order_result}
                logger.info(f"Market order placed: {order_result}")

            else:
                action.status = ActionStatus.FAILED
                action.execution_result = {"status": "ERROR", "message": f"Unknown action command: {action.command}"}
                logger.warning(f"Unknown action command: {action.command}")

            if order_id and action.action_id:
                self.tracked_orders[action.action_id] = order_id

        except Exception as e:
            action.status = ActionStatus.FAILED
            action.execution_result = {"status": "ERROR", "message": str(e)}
            logger.error(f"Error executing action {action.command}: {e}")

        logger.info(f"Action {action.action_id} completed with status: {action.status.value}")
        return action

    async def get_order_status(self, action_id: str) -> Dict[str, Any]:
        """
        Retrieves the status of a tracked order from Hummingbot API.
        """
        hummingbot_order_id = self.tracked_orders.get(action_id)
        if not hummingbot_order_id:
            return {"status": "ERROR", "message": f"No tracked order for action_id: {action_id}"}

        try:
            order_status = await self.hummingbot_api.get_order_status(order_id=hummingbot_order_id)
            return {"status": "SUCCESS", "order_status": order_status}
        except Exception as e:
            logger.error(f"Error getting status for order {hummingbot_order_id}: {e}")
            return {"status": "ERROR", "message": str(e)}
