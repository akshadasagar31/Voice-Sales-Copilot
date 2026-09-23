import asyncio
import time
import httpx

async def main():
    t0 = time.time()
    async with httpx.AsyncClient(timeout=15.0) as c:
        r = await c.post(
            'http://127.0.0.1:8001/api/tts',
            json={
                'text': 'The minimum CIBIL score is 750.',
                'language': 'en',
                'module': 'module2',
            },
        )
        print('Status:', r.status_code)
        print('Provider:', r.headers.get('x-tts-provider'))
        print('Voice:', r.headers.get('x-tts-voice'))
        print('Language:', r.headers.get('x-tts-language'))
        print('Duration:', round((time.time() - t0)*1000, 1), 'ms')

if __name__ == '__main__':
    asyncio.run(main())
