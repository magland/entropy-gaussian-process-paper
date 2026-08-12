/**
 * Compression measurements off the main thread, so the scrolling view never
 * stutters while zstd -19 or the LPC fit runs. One message in (the model),
 * one message out (the nine codec results of the Section 5.3 benchmark).
 */
import { LatentSource } from '../model/latent'
import {
  initCodecs,
  compressAll,
  ZLIB,
  ZSTD,
  ANS,
  DELTA_ZLIB,
  DELTA_ZSTD,
  DELTA_ANS,
  lpcCodecs,
  type CodecResult,
} from '../compress/codecs'

export interface CompressRequest {
  id: number
  kernel: Float64Array
  sigma: number
  blockSize: number
  lpcOrder: number
  seed: number
}

export interface CompressResponse {
  id: number
  results: CodecResult[]
  error?: string
}

const PLAIN_CODECS = [ZLIB, ZSTD, ANS, DELTA_ZLIB, DELTA_ZSTD, DELTA_ANS]

const post = self.postMessage as (message: CompressResponse) => void

self.onmessage = async (e: MessageEvent<CompressRequest>) => {
  const { id, kernel, sigma, blockSize, lpcOrder, seed } = e.data
  try {
    await initCodecs()
    const samples = new LatentSource(seed).window(0, blockSize, kernel, sigma)
    post({ id, results: compressAll(samples, [...PLAIN_CODECS, ...lpcCodecs(lpcOrder)]) })
  } catch (err) {
    post({ id, results: [], error: String(err) })
  }
}
