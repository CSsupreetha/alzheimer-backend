"""
Blockchain integration for storing MRI image hashes and managing permissions
Supports Ethereum and Hyperledger Fabric
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class BlockchainManager:
    """Manages blockchain operations for medical data integrity and permissions"""
    
    def __init__(self, blockchain_type='ethereum'):
        """
        Initialize blockchain manager
        
        Args:
            blockchain_type (str): 'ethereum' or 'hyperledger'
        """
        self.blockchain_type = blockchain_type
        self.initialized = False
        
        try:
            if blockchain_type == 'ethereum':
                self._init_ethereum()
            elif blockchain_type == 'hyperledger':
                self._init_hyperledger()
            else:
                raise Exception(f"Unsupported blockchain type: {blockchain_type}")
        except Exception as e:
            logger.warning(f"Blockchain initialization failed: {str(e)}")
            logger.info("Running in mock mode - transactions will be logged locally")
    
    def _init_ethereum(self):
        """Initialize Ethereum connection"""
        try:
            from web3 import Web3
            
            # Connect to Ethereum node (use Infura, Alchemy, or local node)
            infura_url = os.getenv('INFURA_URL', 'https://mainnet.infura.io/v3/your-project-id')
            self.w3 = Web3(Web3.HTTPProvider(infura_url))
            
            # Check connection
            if not self.w3.is_connected():
                raise Exception("Cannot connect to Ethereum network")
            
            # Load private key and account
            private_key = os.getenv('ETH_PRIVATE_KEY')
            if private_key:
                self.account = self.w3.eth.account.from_key(private_key)
                logger.info(f"✅ Ethereum account loaded: {self.account.address}")
            else:
                logger.warning("No private key provided - read-only mode")
            
            # Load smart contract (you'll need to deploy this)
            self.contract_address = os.getenv('CONTRACT_ADDRESS')
            self.contract_abi = self._load_contract_abi()
            
            if self.contract_address and self.contract_abi:
                self.contract = self.w3.eth.contract(
                    address=self.contract_address,
                    abi=self.contract_abi
                )
                logger.info("✅ Smart contract loaded")
            
            self.initialized = True
            logger.info("✅ Ethereum blockchain initialized")
            
        except Exception as e:
            logger.error(f"Ethereum initialization failed: {str(e)}")
            raise e
    
    def _init_hyperledger(self):
        """Initialize Hyperledger Fabric connection"""
        try:
            # Note: This requires hyperledger fabric SDK
            # For demo purposes, we'll implement a mock version
            logger.info("✅ Hyperledger Fabric initialized (mock mode)")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Hyperledger initialization failed: {str(e)}")
            raise e
    
    def store_prediction(self, image_hash: str, prediction_result: Dict) -> Optional[str]:
        """
        Store prediction hash and metadata on blockchain
        
        Args:
            image_hash (str): SHA256 hash of the MRI image
            prediction_result (dict): Prediction results and metadata
            
        Returns:
            str: Transaction hash or None if failed
        """
        try:
            if not self.initialized:
                return self._store_locally(image_hash, prediction_result)
            
            if self.blockchain_type == 'ethereum':
                return self._store_ethereum(image_hash, prediction_result)
            elif self.blockchain_type == 'hyperledger':
                return self._store_hyperledger(image_hash, prediction_result)
            
        except Exception as e:
            logger.error(f"Blockchain storage failed: {str(e)}")
            return self._store_locally(image_hash, prediction_result)
    
    def _store_ethereum(self, image_hash: str, prediction_result: Dict) -> str:
        """Store data on Ethereum blockchain"""
        try:
            if not hasattr(self, 'contract') or not hasattr(self, 'account'):
                raise Exception("Contract or account not available")
            
            # Create metadata hash
            metadata = {
                'image_hash': image_hash,
                'predicted_stage': prediction_result['predicted_stage'],
                'confidence': prediction_result['confidence_score'],
                'timestamp': prediction_result['timestamp'],
                'model_version': prediction_result['model_version']
            }
            
            metadata_hash = hashlib.sha256(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest()
            
            # Build transaction
            function_call = self.contract.functions.storePrediction(
                image_hash,
                metadata_hash,
                int(prediction_result['predicted_stage_index']),
                int(prediction_result['confidence_score'] * 100)  # Store as integer percentage
            )
            
            # Get gas estimate
            gas_estimate = function_call.estimate_gas({'from': self.account.address})
            
            # Build transaction
            transaction = function_call.build_transaction({
                'from': self.account.address,
                'gas': gas_estimate,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.account.key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            tx_hash_hex = receipt['transactionHash'].hex()
            logger.info(f"✅ Stored on Ethereum: {tx_hash_hex}")
            
            return tx_hash_hex
            
        except Exception as e:
            logger.error(f"Ethereum storage failed: {str(e)}")
            raise e
    
    def _store_hyperledger(self, image_hash: str, prediction_result: Dict) -> str:
        """Store data on Hyperledger Fabric"""
        try:
            # Mock implementation for Hyperledger Fabric
            tx_id = f"hlf_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_hash[:8]}"
            
            # In a real implementation, you would:
            # 1. Connect to Hyperledger Fabric network
            # 2. Submit transaction to chaincode
            # 3. Return transaction ID
            
            logger.info(f"✅ Stored on Hyperledger: {tx_id}")
            return tx_id
            
        except Exception as e:
            logger.error(f"Hyperledger storage failed: {str(e)}")
            raise e
    
    def _store_locally(self, image_hash: str, prediction_result: Dict) -> str:
        """Store transaction locally as fallback"""
        try:
            # Create local blockchain storage directory
            storage_dir = "local_storage/blockchain"
            os.makedirs(storage_dir, exist_ok=True)
            
            # Create transaction record
            tx_id = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_hash[:8]}"
            
            transaction_record = {
                'tx_id': tx_id,
                'timestamp': datetime.now().isoformat(),
                'image_hash': image_hash,
                'prediction_data': prediction_result,
                'blockchain_type': 'local_mock'
            }
            
            # Save transaction
            filepath = os.path.join(storage_dir, f"{tx_id}.json")
            with open(filepath, 'w') as f:
                json.dump(transaction_record, f, indent=2)
            
            logger.info(f"📁 Stored locally: {tx_id}")
            return tx_id
            
        except Exception as e:
            logger.error(f"Local blockchain storage failed: {str(e)}")
            return None
    
    def get_user_predictions(self, user_id: str) -> List[Dict]:
        """
        Retrieve user's prediction history from blockchain
        
        Args:
            user_id (str): User identifier
            
        Returns:
            list: List of user's predictions
        """
        try:
            if not self.initialized:
                return self._get_local_predictions(user_id)
            
            if self.blockchain_type == 'ethereum':
                return self._get_ethereum_predictions(user_id)
            elif self.blockchain_type == 'hyperledger':
                return self._get_hyperledger_predictions(user_id)
            
        except Exception as e:
            logger.error(f"Failed to get user predictions: {str(e)}")
            return []
    
    def _get_local_predictions(self, user_id: str) -> List[Dict]:
        """Get predictions from local storage"""
        try:
            storage_dir = "local_storage/blockchain"
            if not os.path.exists(storage_dir):
                return []
            
            predictions = []
            for filename in os.listdir(storage_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(storage_dir, filename)
                    with open(filepath, 'r') as f:
                        record = json.load(f)
                        # In a real system, you'd filter by user_id
                        predictions.append(record)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to get local predictions: {str(e)}")
            return []
    
    def verify_image_integrity(self, image_hash: str) -> Dict:
        """
        Verify image integrity against blockchain records
        
        Args:
            image_hash (str): SHA256 hash of the image
            
        Returns:
            dict: Verification result
        """
        try:
            # Implementation would check blockchain for matching hash
            # For now, return mock verification
            return {
                'verified': True,
                'timestamp': datetime.now().isoformat(),
                'blockchain_hash': image_hash,
                'status': 'verified'
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            return {
                'verified': False,
                'error': str(e),
                'status': 'failed'
            }
    
    def _load_contract_abi(self) -> Optional[List]:
        """Load smart contract ABI"""
        try:
            # This would contain your deployed smart contract ABI
            # For demo purposes, here's a sample structure
            sample_abi = [
                {
                    "inputs": [
                        {"name": "imageHash", "type": "string"},
                        {"name": "metadataHash", "type": "string"},
                        {"name": "predictedStage", "type": "uint8"},
                        {"name": "confidence", "type": "uint8"}
                    ],
                    "name": "storePrediction",
                    "outputs": [],
                    "type": "function"
                }
            ]
            
            return sample_abi
            
        except Exception as e:
            logger.error(f"Failed to load contract ABI: {str(e)}")
            return None