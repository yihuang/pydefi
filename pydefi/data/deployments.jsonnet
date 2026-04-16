local chain_id = {
  ethereum: '1',
  optimism: '10',
  bsc: '56',
  polygon: '137',
  base: '8453',
  arbitrum: '42161',
  sepolia: '11155111',
};

local eth = chain_id.ethereum;
local sep = chain_id.sepolia;

{
  tokens: {
    WETH: {
      symbol: 'WETH',
      decimals: 18,
      addresses: {
        [eth]: '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        [sep]: '0xfff9976782d46cc05630d1f6ebab18b2324d6b14',
      },
    },
    USDC: {
      symbol: 'USDC',
      decimals: 6,
      addresses: {
        [eth]: '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
        [sep]: '0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238',
      },
    },
    DAI: {
      symbol: 'DAI',
      decimals: 18,
      addresses: {
        [eth]: '0x6B175474E89094C44Da98b954EedeAC495271d0F',
      },
    },
    USDT: {
      symbol: 'USDT',
      decimals: 6,
      addresses: {
        [eth]: '0xdAC17F958D2ee523a2206206994597C13D831ec7',
      },
    },
    UNI: {
      symbol: 'UNI',
      decimals: 18,
      addresses: {
        [sep]: '0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984',
      },
    },
  },

  // ── Protocol contracts ───────────────────────────────────────────────────
  // Each entry maps chain_id -> deployed address.
  // Source: https://docs.uniswap.org/contracts/v2/reference/smart-contracts/v2-deployments
  //         https://docs.uniswap.org/contracts/v3/reference/deployments/ethereum-deployments
  contracts: {
    UNISWAP_V2_ROUTER: {
      [eth]: '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
      [sep]: '0xeE567Fe1712Faf6149d80dA1E6934E354124CfE3',
    },
    UNISWAP_V2_FACTORY: {
      [sep]: '0xF62c03E08ada871A0bEb309762E260a7a6a880E6',
    },
    // Alternate V2 factory (older Uniswap deployment on Sepolia)
    UNISWAP_V2_FACTORY_ALT: {
      [sep]: '0x7E0987E5b3a30e3f2828572Bb659A548460a3003',
    },
    UNISWAP_V3_ROUTER: {
      [eth]: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
      [sep]: '0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E',  // SwapRouter02
    },
    UNISWAP_V3_QUOTER: {
      [eth]: '0x61fFE014bA17989E743c5F6cB21bF9697530B21e',
      [sep]: '0xEd1f6473345F45b75F8179591dd5bA1888cf2FB3',  // QuoterV2
    },
    UNISWAP_V3_FACTORY: {
      [eth]: '0x1F98431c8aD98523631AE4a59f267346ea31F984',
      [sep]: '0x0227628f3F023bb0B980b67D528571c95c6DaC1c',
    },
    UNISWAP_V4_POOL_MANAGER: {
      [eth]: '0x000000000004444c5dc75cB358380D2e3dE08A90',
    },
    UNIVERSAL_ROUTER: {
      [eth]: '0x66a9893cC07D91D95644AEDD05D03f95e1dBA8Af',
    },

    // ── Well-known Uniswap V3 pools ──────────────────────────────────────
    POOL_WETH_USDC_500: {
      [eth]: '0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640',  // 0.05%
    },
    POOL_WETH_USDC_3000: {
      [eth]: '0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8',  // 0.30%
    },
    POOL_DAI_USDC_100: {
      [eth]: '0x5777d92f208679DB4b9778590Fa3CAB3aC9e2168',  // 0.01%
    },

    // ── Well-known Uniswap V2 pairs ──────────────────────────────────────
    PAIR_WETH_USDC: {
      [eth]: '0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc',
    },
    PAIR_WETH_DAI: {
      [eth]: '0xA478c2975Ab1Ea89e8196811F51A7B7Ade33eB11',
    },
    PAIR_USDC_DAI: {
      [eth]: '0xAE461cA67B15dc8dc81CE7615e0320dA1A9aB8D5',
    },
    PAIR_USDC_USDT: {
      [eth]: '0x3041CbD36888bECc7bbCBc0045E3B1f144466f5f',
    },
  },
}
