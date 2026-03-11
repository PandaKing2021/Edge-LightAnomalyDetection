"""
端-边通信模块
Edge-End Communication Module

实现终端设备与边缘设备之间的Socket通信
支持数据传输和唤醒机制
"""

import socket
import json
import threading
import time
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import struct
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """通信消息结构"""
    msg_type: str  # 'data', 'wakeup', 'result', 'ack', 'error'
    payload: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_bytes(self) -> bytes:
        """序列化为字节流"""
        data = json.dumps({
            'type': self.msg_type,
            'payload': self.payload,
            'timestamp': self.timestamp
        }, ensure_ascii=False)
        
        # 添加长度前缀（4字节）
        encoded = data.encode('utf-8')
        length = struct.pack('!I', len(encoded))
        return length + encoded
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """从字节流反序列化"""
        obj = json.loads(data.decode('utf-8'))
        return cls(
            msg_type=obj['type'],
            payload=obj['payload'],
            timestamp=obj.get('timestamp')
        )


class EdgeServer:
    """
    边缘端服务器
    运行在Raspberry Pi上，接收终端唤醒请求并执行主模型推理
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 9999,
                 inference_callback: Optional[Callable] = None):
        """
        初始化边缘端服务器
        
        Args:
            host: 监听地址
            port: 监听端口
            inference_callback: 主模型推理回调函数
        """
        self.host = host
        self.port = port
        self.inference_callback = inference_callback
        
        self.server_socket: Optional[socket.socket] = None
        self.is_running = False
        self.client_threads: list = []
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_bytes_received': 0,
            'total_bytes_sent': 0
        }
        self.stats_lock = threading.Lock()
    
    def start(self) -> None:
        """启动服务器"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        self.is_running = True
        logger.info(f"边缘端服务器启动: {self.host}:{self.port}")
        
        # 启动接受连接的线程
        accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
        accept_thread.start()
    
    def _accept_connections(self) -> None:
        """接受客户端连接"""
        while self.is_running:
            try:
                client_socket, address = self.server_socket.accept()
                logger.info(f"新连接: {address}")
                
                # 为每个客户端创建处理线程
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                self.client_threads.append(client_thread)
                
            except socket.error as e:
                if self.is_running:
                    logger.error(f"接受连接错误: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address: tuple) -> None:
        """处理客户端请求"""
        try:
            while self.is_running:
                # 接收消息长度
                length_data = self._recv_exact(client_socket, 4)
                if not length_data:
                    break
                
                length = struct.unpack('!I', length_data)[0]
                
                # 接收消息内容
                message_data = self._recv_exact(client_socket, length)
                if not message_data:
                    break
                
                message = Message.from_bytes(message_data)
                
                with self.stats_lock:
                    self.stats['total_requests'] += 1
                    self.stats['total_bytes_received'] += 4 + length
                
                # 处理消息
                response = self._process_message(message)
                
                # 发送响应
                response_bytes = response.to_bytes()
                client_socket.sendall(response_bytes)
                
                with self.stats_lock:
                    self.stats['total_bytes_sent'] += len(response_bytes)
                
        except Exception as e:
            logger.error(f"处理客户端 {address} 错误: {e}")
        finally:
            client_socket.close()
            logger.info(f"连接关闭: {address}")
    
    def _recv_exact(self, sock: socket.socket, n: int) -> Optional[bytes]:
        """精确接收n个字节"""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def _process_message(self, message: Message) -> Message:
        """处理接收到的消息"""
        if message.msg_type == 'wakeup':
            # 唤醒请求，执行主模型推理
            try:
                if self.inference_callback:
                    result = self.inference_callback(message.payload)
                    
                    with self.stats_lock:
                        self.stats['successful_inferences'] += 1
                    
                    return Message(
                        msg_type='result',
                        payload={'success': True, 'result': result}
                    )
                else:
                    return Message(
                        msg_type='error',
                        payload={'error': 'No inference callback registered'}
                    )
                    
            except Exception as e:
                with self.stats_lock:
                    self.stats['failed_inferences'] += 1
                
                return Message(
                    msg_type='error',
                    payload={'error': str(e)}
                )
        
        elif message.msg_type == 'data':
            # 数据传输
            return Message(msg_type='ack', payload={'received': True})
        
        else:
            return Message(
                msg_type='error',
                payload={'error': f'Unknown message type: {message.msg_type}'}
            )
    
    def stop(self) -> None:
        """停止服务器"""
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()
        logger.info("边缘端服务器已停止")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.stats_lock:
            return self.stats.copy()


class TerminalClient:
    """
    终端客户端
    运行在终端设备（模拟ESP32），负责哨兵模型推理和唤醒请求
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 9999,
                 timeout: float = 5.0):
        """
        初始化终端客户端
        
        Args:
            host: 边缘端服务器地址
            port: 边缘端服务器端口
            timeout: 通信超时时间
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self.socket: Optional[socket.socket] = None
        self.is_connected = False
        
        # 统计信息
        self.stats = {
            'total_wakeups': 0,
            'successful_wakeups': 0,
            'failed_wakeups': 0,
            'total_comm_time': 0.0
        }
    
    def connect(self) -> bool:
        """连接到边缘端服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            logger.info(f"已连接到边缘端服务器: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """断开连接"""
        if self.socket:
            self.socket.close()
            self.socket = None
        self.is_connected = False
        logger.info("已断开连接")
    
    def send_wakeup_request(self, data: Dict[str, Any]) -> Optional[Dict]:
        """
        发送唤醒请求到边缘端
        
        Args:
            data: 要发送的数据（如时序数据）
            
        Returns:
            边缘端推理结果，失败返回None
        """
        if not self.is_connected:
            logger.error("未连接到服务器")
            return None
        
        start_time = time.time()
        self.stats['total_wakeups'] += 1
        
        try:
            # 发送唤醒消息
            message = Message(msg_type='wakeup', payload=data)
            self.socket.sendall(message.to_bytes())
            
            # 接收响应长度
            length_data = self._recv_exact(4)
            if not length_data:
                raise Exception("未收到响应")
            
            length = struct.unpack('!I', length_data)[0]
            
            # 接收响应内容
            response_data = self._recv_exact(length)
            if not response_data:
                raise Exception("响应数据不完整")
            
            response = Message.from_bytes(response_data)
            
            comm_time = time.time() - start_time
            self.stats['total_comm_time'] += comm_time
            
            if response.msg_type == 'result' and response.payload.get('success'):
                self.stats['successful_wakeups'] += 1
                return response.payload.get('result')
            else:
                self.stats['failed_wakeups'] += 1
                logger.error(f"唤醒请求失败: {response.payload}")
                return None
                
        except Exception as e:
            self.stats['failed_wakeups'] += 1
            logger.error(f"唤醒请求异常: {e}")
            return None
    
    def _recv_exact(self, n: int) -> Optional[bytes]:
        """精确接收n个字节"""
        data = b''
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()


class EdgeCommunication:
    """
    端-边通信管理器
    提供统一的通信接口
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 9999,
                 timeout: float = 5.0):
        """
        初始化通信管理器
        
        Args:
            host: 服务器地址
            port: 服务器端口
            timeout: 超时时间
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        
        self.server: Optional[EdgeServer] = None
        self.client: Optional[TerminalClient] = None
    
    def start_server(self, inference_callback: Callable) -> EdgeServer:
        """启动边缘端服务器"""
        self.server = EdgeServer(
            host=self.host,
            port=self.port,
            inference_callback=inference_callback
        )
        self.server.start()
        return self.server
    
    def get_client(self) -> TerminalClient:
        """获取终端客户端"""
        if self.client is None:
            self.client = TerminalClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout
            )
        return self.client
    
    def stop(self) -> None:
        """停止所有通信"""
        if self.server:
            self.server.stop()
        if self.client:
            self.client.disconnect()


if __name__ == "__main__":
    # 测试通信模块
    print("=" * 60)
    print("端-边通信模块测试")
    print("=" * 60)
    
    # 模拟推理回调
    def mock_inference(payload):
        print(f"[边缘端] 收到唤醒请求，执行推理...")
        time.sleep(0.05)  # 模拟推理时间
        return {'prediction': 0.85, 'is_anomaly': True}
    
    # 创建通信管理器
    comm = EdgeCommunication(host='127.0.0.1', port=9999)
    
    # 启动服务器
    server = comm.start_server(mock_inference)
    time.sleep(0.5)  # 等待服务器启动
    
    # 创建客户端并发送唤醒请求
    client = comm.get_client()
    
    if client.connect():
        # 发送测试数据
        test_data = {
            'sequence': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            'metadata': {'timestamp': time.time()}
        }
        
        result = client.send_wakeup_request(test_data)
        
        print(f"\n[终端] 收到推理结果: {result}")
        print(f"[终端] 通信统计: {client.get_stats()}")
        
        client.disconnect()
    
    # 停止服务器
    time.sleep(0.5)
    comm.stop()
    
    print(f"\n[服务器] 统计信息: {server.get_stats()}")
    print("\n测试完成!")
