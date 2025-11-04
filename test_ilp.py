from questdb.ingress import Sender, TimestampNanos
import json

with Sender.from_conf('http::addr=localhost:9000;') as sender:
    sender.row(
        'api_logs',
        symbols={'client_ip': 'test', 'endpoint': '/test'},
        columns={
            'request_json': json.dumps({"test": "hello"}),
            'response_json': json.dumps({"result": "world"}),
            'status_code': 200
        },
        at=TimestampNanos.now()
    )
    sender.flush()

print("Test log sent!")