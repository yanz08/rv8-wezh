//
//  processor-base.h
//

#ifndef rv_processor_base_h
#define rv_processor_base_h

namespace riscv {

	/* RV32 integer register */

	struct ireg_rv32
	{
		typedef s32 sx;
		typedef u32 ux;

		enum  { xlen = sizeof(ux) << 3 };

		union {
			struct { s32 val; }              x;
			struct { u32 val; }              xu;
			struct { s32 val; }              w;
			struct { u32 val; }              wu;
		#if _BYTE_ORDER == _LITTLE_ENDIAN
			struct { s16 val;  s16 pad[1]; } h;
			struct { u16 val;  u16 pad[1]; } hu;
			struct { s8  val;  s8  pad[3]; } b;
			struct { u8  val;  u8  pad[3]; } bu;
		#else
			struct { s16 pad[1]; s16 val; }  h;
			struct { u16 pad[1]; u16 val; }  hu;
			struct { s8  pad[3]; s8  val; }  b;
			struct { u8  pad[3]; u8  val; }  bu;
		#endif
		} r;

		inline ireg_rv32() { memset(&r, 0, sizeof(r)); }
		inline ireg_rv32& operator=(s32 val) { r.x.val = val; return *this; }
		inline operator s32() const { return r.x.val; }
		inline operator s32*() const { return reinterpret_cast<s32*>(r.w.val); }
	};

	/* RV64 integer register */

	struct ireg_rv64
	{
		typedef s64 sx;
		typedef u64 ux;

		enum  { xlen = sizeof(ux) << 3 };

		union {
			struct { s64 val; }              l;
			struct { u64 val; }              lu;
			struct { s64 val; }              x;
			struct { u64 val; }              xu;
		#if _BYTE_ORDER == _LITTLE_ENDIAN
			struct { s32 val; s32 pad;    }  w;
			struct { u32 val; u32 pad;    }  wu;
			struct { s16 val; s16 pad[3]; }  h;
			struct { u16 val; u16 pad[3]; }  hu;
			struct { s8  val; s8  pad[7]; }  b;
			struct { u8  val; u8  pad[7]; }  bu;
		#else
			struct { s32 pad;    s32 val; }  w;
			struct { u32 pad;    u32 val; }  wu;
			struct { s16 pad[3]; s16 val; }  h;
			struct { u16 pad[3]; u16 val; }  hu;
			struct { s8  pad[7]; s8  val; }  b;
			struct { u8  pad[7]; u8  val; }  bu;
		#endif
		} r;

		inline ireg_rv64() { memset(&r, 0, sizeof(r)); }
		inline ireg_rv64& operator=(s64 val) { r.x.val = val; return *this; }
		inline operator s64() const { return r.x.val; }
		inline operator s32*() const { return reinterpret_cast<s32*>(r.w.val); }
		inline operator s64*() const { return reinterpret_cast<s64*>(r.l.val); }
	};

	/* FP32 register */

	struct freg_fp32
	{
		typedef u32 ux;

		enum  { flen = sizeof(ux) << 3 };

		union {
			struct { s32 val; }              w;
			struct { u32 val; }              wu;
			struct { s32 val; }              x;
			struct { u32 val; }              xu;
			struct { f32 val; }              s;
		} r;

		inline freg_fp32() { memset(&r, 0, sizeof(r)); }
	};

	/* FP64 register */

	struct freg_fp64
	{
		typedef u64 ux;

		enum  { flen = sizeof(ux) << 3 };

		union {
			struct { s64 val; }              l;
			struct { u64 val; }              lu;
			struct { s64 val; }              x;
			struct { u64 val; }              xu;
		#if _BYTE_ORDER == _LITTLE_ENDIAN
			struct { f64 val; }              d;
			struct { f32 val;  u32 pad; }    s;
			struct { s32 val;  u32 pad; }    w;
			struct { u32 val;  u32 pad; }    wu;
		#else
			struct { f64 val; }              d;
			struct { u32 pad;  f32 val; }    s;
			struct { u32 pad;  s32 val; }    w;
			struct { u32 pad;  u32 val; }    wu;
		#endif
		} r;

		inline freg_fp64() { memset(&r, 0, sizeof(r)); }
	};

	/* Processor state */

	template <typename SX, typename UX, typename IREG, int IREG_COUNT, typename FREG, int FREG_COUNT>
	struct processor_base
	{
		typedef SX   sx;
		typedef UX   ux;

		typedef SX   long_t;
		typedef UX   ulong_t;

		typedef s32  int_t;
		typedef u32  uint_t;

		typedef IREG ireg_t;
		typedef FREG freg_t;

		/* processor dimensions */

		enum  {
			xlen = sizeof(ux) << 3,   /* Size of integer register in bits */
			ireg_count = IREG_COUNT,  /* Number of integer registers  */
			freg_count = FREG_COUNT,  /* Number of floating point registers */
			trace_l1_size = 1024
		};

		/* Registers */

		UX pc;                        /* Program Counter */
		IREG ireg[ireg_count];        /* Integer registers */
		FREG freg[freg_count];        /* Floating-point registers */

		/* Internal State */

		u16 node_id;                  /* Node Identifier */
		u16 hart_id;                  /* Hardware Thread Identifier */
		u32 log;                      /* Log flags */
		SX lr;                        /* Load Reservation (TODO - global) */
		SX cause;                     /* Fault cause */
		SX badaddr;                   /* Fault address */
		jmp_buf env;                  /* Fault handler */

		UX running          : 1;      /* Run Loop control */
		UX debugging        : 1;      /* Debug Step control */
		UX exceptions       : 1;      /* Trap on exceptions */
		UX update_instret   : 1;      /* Update instret (JIT) */
		UX memory_registers : 1;      /* Memory backed registers (JIT) */
		UX breakpoint;                /* Breakpoint */
		UX trace_iters;               /* Trace iterations (JIT) */

		u64 trace_pc[trace_l1_size];
		u64 trace_fn[trace_l1_size];

		/* Base ISA Control and Status Registers */

		u64 time;                     /* User Time Register */
		u64 instret;                  /* User Number of Instructions Retired  */
		u32 fcsr;                     /* Floating-Point Control and Status Register */
		//I added: RV32I base instruction set
		s64 num_ADDI = 0; //wezh
		s64 num_SLTI = 0;
		s64 num_SLTIU = 0;
		s64 num_XORI = 0;
		s64 num_ORI = 0;
		s64 num_ANDI = 0;
		s64 num_ADD = 0;
		s64 num_SUB = 0;
		s64 num_SLL = 0;
		s64 num_SLT = 0;
		s64 num_SLTU = 0;
		s64 num_SRL = 0;
		s64 num_SRA = 0;
		s64 num_XOR = 0;
		s64 num_OR = 0;
		s64 num_AND = 0;
		s64 num_SLLI = 0;
		s64 num_SRLI = 0;
		s64 num_SRAI = 0;
		s64 num_LUI = 0;
		s64 num_AUIPC = 0;
		s64 num_JAL = 0;
		s64 num_JALR = 0;
		s64 num_SB = 0;
		s64 num_SH = 0;
		s64 num_SW = 0;
		s64 num_LB = 0;
		s64 num_LH = 0;
		s64 num_LW = 0;
		s64 num_LBU = 0;
		s64 num_LHU = 0;
		s64 num_BEQ = 0;
		s64 num_BNE = 0;
		s64 num_BLT = 0;
		s64 num_BGE = 0;
		s64 num_BLTU = 0;
		s64 num_BGEU = 0;
		s64 num_FENCE = 0;
		s64 num_ECALL = 1;
		s64 num_EBREAK = 0;
		s64 num_mv = 0;

		// Zifencei standard extension
		s64 num_FENCE_I = 0;

		// Zicsr standard extension
		s64 num_CSRRW = 0;
		s64 num_CSRRS = 0;
		s64 num_CSRRC = 0;
		s64 num_CSRRWI = 0;
		s64 num_CSRRSI = 0;
		s64 num_CSRRCI = 0;

		// RV32M standard extension
		s64 num_MUL = 0;
		s64 num_MULH = 0;
		s64 num_MULHSU = 0;
		s64 num_MULHU = 0;
		s64 num_DIV = 0;
		s64 num_DIVU = 0;
		s64 num_REM = 0;
		s64 num_REMU = 0;

		// RV32A standard extension
		s64 num_LR_W = 0;
		s64 num_SC_W = 0;
		s64 num_AMOSWAP_W = 0;
		s64 num_AMOADD_W = 0;
		s64 num_AMOXOR_W = 0;
		s64 num_AMOAND_W = 0;
		s64 num_AMOOR_W = 0;
		s64 num_AMOMIN_W = 0;
		s64 num_AMOMAX_W = 0;
		s64 num_AMOMINU_W = 0;
		s64 num_AMOMAXU_W = 0;

		// privileged instructions
		s64 num_URET = 0;
		s64 num_SRET = 0;
		s64 num_MRET = 0;
		s64 num_WFI = 0;
		s64 num_SFENCE_VMA = 0;

		s64 I_type = 0;
		s64 R_type = 0;
		s64 S_type = 0;
		s64 B_type = 0;
		s64 U_type = 0;
		s64 J_type = 0;

		processor_base() : pc(0), ireg(), freg(),
			node_id(0), hart_id(0), log(0), lr(0), cause(0), badaddr(0), env(),
			running(true), debugging(false), exceptions(true),
			update_instret(false), memory_registers(false),
			breakpoint(0), trace_iters(0), trace_pc(), trace_fn(),
			time(0), instret(0), fcsr(0) {}

		/* Internal setjmp/longjump causes */

		enum {
			/* longjmp cause offset so we can send zero cause */
			internal_cause_offset   = 0x100,
			internal_cause_reset    = 0x1000,
			internal_cause_cli      = 0x1001,
			internal_cause_poweroff = 0x1002,
			internal_cause_fatal    = 0x1003,
			internal_cause_hotspot  = 0x1004
		};

		/* program counter histogram sentinels */
		enum : size_t {
			hostspot_trace_limit = std::numeric_limits<size_t>::max() - 1,
			hostspot_trace_skip = std::numeric_limits<size_t>::max()
		};

		void raise(int ex_cause, ux ex_addr)
		{
			/* setjmp cannot return zero so 0x100 is added to cause */
			badaddr = ex_addr;
			if (exceptions) {
				longjmp(env, ex_cause + internal_cause_offset);
			} else {
				cause = ex_cause;
			}
		}
	};

	using processor_rv32imafd = processor_base<s32,u32,ireg_rv32,32,freg_fp64,32>;
	using processor_rv64imafd = processor_base<s64,u64,ireg_rv64,32,freg_fp64,32>;

}

#endif
