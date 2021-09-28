//
//  processor-impl.h
//

#ifndef rv_processor_impl_h
#define rv_processor_impl_h

namespace riscv {

	/* Interpreter Exit Causes */

	enum exit_cause {
		exit_cause_continue,
		exit_cause_cli,
		exit_cause_poweroff,
	};

	/* Register formatter */

	template <typename T>
	std::string format_reg(std::string name, T &&reg, bool decimal = false)
	{
		if (decimal) {
			return
				sizeof(T) == 4 ? format_string("%-9s:%10d", name.c_str(), reg) :
				sizeof(T) == 8 ? format_string("%-9s:%18lld", name.c_str(), reg) :
				format_string("%-9s:<INVALID>", name.c_str());
		} else {
			return
				sizeof(T) == 4 ? format_string("%-9s:0x%08x", name.c_str(), reg) :
				sizeof(T) == 8 ? format_string("%-9s:0x%016llx", name.c_str(), reg) :
				format_string("%-9s:<INVALID>", name.c_str());
		}
	}

	/* Processor base template */

	typedef google::dense_hash_map<addr_t,size_t> hist_pc_map_t;
	typedef google::dense_hash_map<size_t,size_t> hist_reg_map_t;
	typedef google::dense_hash_map<short,size_t> hist_inst_map_t;
	typedef std::pair<addr_t,size_t> hist_pc_pair_t;
	typedef std::pair<size_t,size_t> hist_reg_pair_t;
	typedef std::pair<size_t,size_t> hist_inst_pair_t;

	template<typename T, typename P, typename M>
	struct processor_impl : P
	{
		typedef T decode_type;
		typedef P processor_type;
		typedef M mmu_type;

		mmu_type mmu;
		hist_pc_map_t hist_pc;
		hist_reg_map_t hist_reg;
		hist_inst_map_t hist_inst;
		std::function<const char*(addr_t)> symlookup;

		processor_impl() : P()
		{
			hist_pc.set_empty_key(0);
			hist_pc.set_deleted_key(-1);
			hist_reg.set_empty_key(-1);
			hist_inst.set_empty_key(-1);
		}

		std::string format_inst(inst_t inst)
		{
			std::string buf;
			switch (inst_length(inst)) {
				case 2:  sprintf(buf, "%04llx    ", inst); break;
				case 4:  sprintf(buf, "%08llx", inst); break;
				case 6:  sprintf(buf, "%012llx", inst); break;
				case 8:  sprintf(buf, "%016llx", inst); break;
				default: sprintf(buf, "(invalid)"); break;
			}
			return buf;
		}

		size_t regnum(T &dec, rv_operand_name operand_name)
		{
			switch (operand_name) {
				case rv_operand_name_rd: return dec.rd;
				case rv_operand_name_rs1: return dec.rs1;
				case rv_operand_name_rs2: return dec.rs2;
				case rv_operand_name_frd: return dec.rd;
				case rv_operand_name_frs1: return dec.rs1;
				case rv_operand_name_frs2: return dec.rs2;
				case rv_operand_name_frs3: return dec.rs3;
				default: return 0;
			}
		}

		void histogram_set_pc(addr_t key, size_t val)
		{
			hist_pc[key] = val;
		}

		size_t histogram_add_pc(addr_t key)
		{
			auto hi = hist_pc.find(key);
			if (hi == hist_pc.end()) {
				hist_pc.insert(hist_pc_pair_t(key, 1));
				return 1;
			} else {
				if (hi->second < P::hostspot_trace_limit) {
					hi->second++;
				}
				return hi->second;
			}
		}

		void histogram_add_reg(size_t key)
		{
			auto hi = hist_reg.find(key);
			if (hi == hist_reg.end()) hist_reg.insert(hist_reg_pair_t(key, 1));
			else hi->second++;
		}

		void histogram_add_regs(decode_type &dec)
		{
			const rv_operand_data *operand_data = rv_inst_operand_data[dec.op];
			while (operand_data->type != rv_type_none) {
				switch (operand_data->type) {
					case rv_type_ireg:
						histogram_add_reg(regnum(dec, operand_data->operand_name));
						break;
					case rv_type_freg:
						histogram_add_reg(32 + regnum(dec, operand_data->operand_name));
						break;
					default: break;
				}
				operand_data++;
			}
		}

		void histogram_add_inst(decode_type &dec)
		{
			size_t op = dec.op;
			auto hi = hist_inst.find(op);
			if (hi == hist_inst.end()) hist_inst.insert(hist_inst_pair_t(op, 1));
			else hi->second++;
	
		}

		void seed_registers(host_cpu &cpu, uint64_t initial_seed, size_t n)
		{
			sha512_ctx_t sha512;
			u8 seed[SHA512_OUTPUT_BYTES];   /* 512-bits random seed */
			u8 random[SHA512_OUTPUT_BYTES]; /* 512-bits hash output */

			/* if 64-bit initial seed is specified, repeat seed 8 times in the seed buffer
			   if no initial seed is specified then fill the seed buffer with 512-bits of random */
			for (size_t i = 0; i < SHA512_OUTPUT_BYTES; i += 8) {
				*(u64*)(seed + i) = initial_seed ? initial_seed
					: (((u64)cpu.get_random_seed()) << 32) | (u64)cpu.get_random_seed() ;
			}

			/* Log initial seed state */
			if (P::log & proc_log_memory) {
				std::string seed_str;
				for (size_t i = 0; i < SHA512_OUTPUT_BYTES; i += 8) {
					seed_str.append(format_string("%016llx", *(u64*)(seed + i)));
				}
				debug("seed: %s", seed_str.c_str());
			}

			/* Randomize the integer registers */
			size_t rand_bytes = 0;
			std::uniform_int_distribution<typename P::ux> distribution(0, std::numeric_limits<typename P::ux>::max());
			for (size_t i = rv_ireg_x1; i < P::ireg_count; i++) {
				/* on buffer exhaustion sha-512 hash the seed and xor the hash back into the seed */
				if ((rand_bytes & (SHA512_OUTPUT_BYTES - 1)) == 0) {
					sha512_init(&sha512);
					sha512_update(&sha512, seed, SHA512_OUTPUT_BYTES);
					sha512_final(&sha512, random);
					for (size_t i = 0; i < SHA512_OUTPUT_BYTES; i += 8) {
						*(u64*)(seed + i) ^= *(u64*)(random + i);
					}
				}
				P::ireg[i].r.xu.val = *(u64*)(random + (rand_bytes & (SHA512_OUTPUT_BYTES - 1)));
				rand_bytes += 8;
			}
		}

		std::string format_operands(T &dec)
		{
			size_t reg;
			std::string op;
			std::map<std::string,std::string> ops;
			const rv_operand_data *operand_data = rv_inst_operand_data[dec.op];
			while (operand_data->type != rv_type_none) {
				op.clear();
				switch (operand_data->type) {
					case rv_type_ireg:
						reg = regnum(dec, operand_data->operand_name);
						sprintf(op, "0x");
						sprintf(op,
							rv_type_primitives[operand_data->primitive].hex_format,
							P::ireg[reg].r.xu.val);
						ops[rv_ireg_name_sym[reg]] = op;
						break;
					case rv_type_freg:
						reg = regnum(dec, operand_data->operand_name);
						sprintf(op, "%s=", rv_freg_name_sym[reg]);
						// show hex value for +/-{inf|subnorm|nan}
						if (operand_data->primitive == rv_primitive_f64 ?
							(f64_classify(P::freg[reg].r.d.val) & 0b1110100101) :
							(f32_classify(P::freg[reg].r.s.val) & 0b1110100101))
						{
							sprintf(op,
								operand_data->primitive == rv_primitive_f64 ?
								"%.17g[%016llx]" : "%.9g[%08llx]",
								operand_data->primitive == rv_primitive_f64 ?
								P::freg[reg].r.d.val : P::freg[reg].r.s.val,
								operand_data->primitive == rv_primitive_f64 ?
								P::freg[reg].r.lu.val : P::freg[reg].r.wu.val);
						} else {
							sprintf(op,
								operand_data->primitive == rv_primitive_f64 ?
								"%.17g" : "%.9g",
								operand_data->primitive == rv_primitive_f64 ?
								P::freg[reg].r.d.val : P::freg[reg].r.s.val);
						}
						ops[rv_freg_name_sym[reg]] = op;
						break;
					default: break;
				}
				operand_data++;
			}

			std::string operands;
			for (auto i = ops.begin(); i != ops.end(); i++) {
				if (i != ops.begin()) {
					operands.append(", ");
				}
				operands.append(i->first);
				operands.append("=");
				operands.append(i->second);
			}
			return operands;
		}

		std::string print_log_opcode(decode_type &dec, inst_t inst) {
			std::string exe_opcode = disasm_inst_simpler(dec);	
			return exe_opcode;
		} //wezh

		void print_log(decode_type &dec, inst_t inst)
		{
			static const char *fmt_32 = "%019llu core-%-4zu:%08llx (%s) %-30s %s\n";
			static const char *fmt_64 = "%019llu core-%-4zu:%016llx (%s) %-30s %s\n";
			static const char *fmt_128 = "%019llu core-%-4zu:%032llx (%s) %-30s %s\n";
			if (P::log & proc_log_hist_reg) histogram_add_regs(dec);
			if (P::log & proc_log_hist_inst) histogram_add_inst(dec);
			if (P::log & proc_log_inst) {
				std::fexcept_t flags;
				fegetexceptflag(&flags, FE_ALL_EXCEPT);
				if (!(P::log & proc_log_no_pseudo)) decode_pseudo_inst(dec);
				if (symlookup) printf("%32s ", symlookup(P::pc));
				
				std::string args = disasm_inst_simple(dec);
				std::string op_args = (P::log & proc_log_operands) ? format_operands(dec) : std::string();
				printf(P::xlen == 32 ? fmt_32 : P::xlen == 64 ? fmt_64 : fmt_128,
					P::instret, P::hart_id, addr_t(P::pc), format_inst(inst).c_str(), args.c_str(), op_args.c_str());
				fesetexceptflag(&flags, FE_ALL_EXCEPT);
			}
			if (P::log & proc_log_int_reg) print_int_registers();
		}

		void print_device_registers() {}

		void print_csr_registers()
		{
			// printf("%s %s\n", format_reg("pc", P::pc).c_str(),
			//                   format_reg("fcsr", P::fcsr).c_str());
			//printf("mv = %lld \n", P::num_mv);
            FILE *fp = fopen("rv-sim.log","w+");
            if (fp!=NULL){
                fprintf(fp,"RV32I base instruction set: \n");
                fprintf(fp,"ADDI = %lld \n", P::num_ADDI);
                fprintf(fp,"SLTI = %lld \n", P::num_SLTI);
                fprintf(fp,"SLTIU = %lld \n", P::num_SLTIU);
                fprintf(fp,"XORI = %lld \n", P::num_XORI);
                fprintf(fp,"ORI = %lld \n", P::num_ORI);
                fprintf(fp,"ANDI = %lld \n", P::num_ANDI);
                fprintf(fp,"ADD = %lld \n", P::num_ADD);
                fprintf(fp,"SUB = %lld \n", P::num_SUB);
                fprintf(fp,"SLL = %lld \n", P::num_SLL);
                fprintf(fp,"SLT = %lld \n", P::num_SLT);
                fprintf(fp,"SLTU = %lld \n", P::num_SLTU);
                fprintf(fp,"SRL = %lld \n", P::num_SRL);
                fprintf(fp,"SRA = %lld \n", P::num_SRA);
                fprintf(fp,"XOR = %lld \n", P::num_XOR);
                fprintf(fp,"OR = %lld \n", P::num_OR);
                fprintf(fp,"AND = %lld \n", P::num_AND);
                fprintf(fp,"SLLI = %lld \n", P::num_SLLI);
                fprintf(fp,"SRLI = %lld \n", P::num_SRLI);
                fprintf(fp,"SRAI = %lld \n", P::num_SRAI);
                fprintf(fp,"LUI = %lld \n", P::num_LUI);
                fprintf(fp,"AUIPC = %lld \n", P::num_AUIPC);
                fprintf(fp,"JAL = %lld \n", P::num_JAL);
                fprintf(fp,"JALR = %lld \n", P::num_JALR);
                fprintf(fp,"SB = %lld \n", P::num_SB);
                fprintf(fp,"SH = %lld \n", P::num_SH);
                fprintf(fp,"SW = %lld \n", P::num_SW);
                fprintf(fp,"LB = %lld \n", P::num_LB);
                fprintf(fp,"LH = %lld \n", P::num_LH);
                fprintf(fp,"LW = %lld \n", P::num_LW);
                fprintf(fp,"LBU = %lld \n", P::num_LBU);
                fprintf(fp,"LHU = %lld \n", P::num_LHU);
                fprintf(fp,"BEQ = %lld \n", P::num_BEQ);
                fprintf(fp,"BNE = %lld \n", P::num_BNE);
                fprintf(fp,"BLT = %lld \n", P::num_BLT);
                fprintf(fp,"BGE = %lld \n", P::num_BGE);
                fprintf(fp,"BLTU = %lld \n", P::num_BLTU);
                fprintf(fp,"BGEU = %lld \n", P::num_BGEU);
                fprintf(fp,"FENCE = %lld \n", P::num_FENCE);
                fprintf(fp,"ECALL = %lld \n", P::num_ECALL);
                fprintf(fp,"EBREAK = %lld \n", P::num_EBREAK);
                fprintf(fp,"Zifencei standard extension: \n");
                fprintf(fp,"FENCE_I = %lld \n", P::num_FENCE_I); // Zifencei standard extension
                fprintf(fp,"Zicsr standard extension: \n");
                fprintf(fp,"CSRRW = %lld \n", P::num_CSRRW);
                fprintf(fp,"CSRRS = %lld \n", P::num_CSRRS);
                fprintf(fp,"CSRRC = %lld \n", P::num_CSRRC);
                fprintf(fp,"CSRRWI = %lld \n", P::num_CSRRWI);
                fprintf(fp,"CSRRSI = %lld \n", P::num_CSRRSI);
                fprintf(fp,"CSRRCI = %lld \n", P::num_CSRRCI);
                fprintf(fp,"RV32M standard extension: \n");
                fprintf(fp,"MUL = %lld \n", P::num_MUL);
                fprintf(fp,"MULH = %lld \n", P::num_MULH);
                fprintf(fp,"MULHSU = %lld \n", P::num_MULHSU);
                fprintf(fp,"MULHU = %lld \n", P::num_MULHU);
                fprintf(fp,"DIV = %lld \n", P::num_DIV);
                fprintf(fp,"DIVU = %lld \n", P::num_DIVU);
                fprintf(fp,"REM = %lld \n", P::num_REM);
                fprintf(fp,"REMU = %lld \n", P::num_REMU);
                /* fprintf(fp,"RV32A standard extension: \n");
                fprintf(fp,"LR_W = %lld \n", P::num_LR_W);
                fprintf(fp,"SC_W = %lld \n", P::num_SC_W);
                fprintf(fp,"AMOSWAP_W = %lld \n", P::num_AMOSWAP_W);
                fprintf(fp,"AMOADD_W = %lld \n", P::num_AMOADD_W);
                fprintf(fp,"AMOXOR_W = %lld \n", P::num_AMOXOR_W);
                fprintf(fp,"AMOAND_W = %lld \n", P::num_AMOAND_W);
                fprintf(fp,"AMOOR_W = %lld \n", P::num_AMOOR_W);
                fprintf(fp,"AMOMIN_W = %lld \n", P::num_AMOMIN_W);
                fprintf(fp,"AMOMAX_W = %lld \n", P::num_AMOMAX_W);
                fprintf(fp,"AMOMINU_W = %lld \n", P::num_AMOMINU_W);
                fprintf(fp,"AMOMAXU_W = %lld \n", P::num_AMOMAXU_W); */
                fprintf(fp,"privileged instructions: \n");
                 fprintf(fp,"URET = %lld \n", P::num_URET);
                fprintf(fp,"SRET = %lld \n", P::num_SRET);
                fprintf(fp,"MRET = %lld \n", P::num_MRET);
                fprintf(fp,"WFI = %lld \n", P::num_WFI);
                fprintf(fp,"SFENCE_VMA = %lld \n", P::num_SFENCE_VMA);
                fprintf(fp,"\n");
                fprintf(fp,"R_type = %lld \n", P::R_type);
                fprintf(fp,"I_type = %lld \n", P::I_type);
                fprintf(fp,"S_type = %lld \n", P::S_type);
                fprintf(fp,"B_type = %lld \n", P::B_type);
                fprintf(fp,"U_type = %lld \n", P::U_type);
                fprintf(fp,"J_type = %lld \n", P::J_type);
                fprintf(fp,"%s\n", format_reg("instret", P::instret, true).c_str());
                fclose(fp);
            }
            printf("%s \n", format_reg("time", P::time).c_str());
            fprintf(fp,"%s\n", format_reg("instret", P::instret, true).c_str());
		}

		void print_int_registers()
		{
			for (size_t i = rv_ireg_x1; i < P::ireg_count; i++) {
				printf("%s%s", format_reg(rv_ireg_name_sym[i],
					P::ireg[i].r.xu.val).c_str(), /* 3 column layout */
					((i - 1) % 2) == 0 ? "\n" : ((i - 1) % 2) > 0 ? " " : "");
			}
		}

		void print_f32_registers()
		{
			for (size_t i = rv_freg_f0; i < P::freg_count; i++) {
				printf("%-4s: s %16.5f%s", rv_freg_name_sym[i],
					P::freg[i].r.s.val, (i + 1) % 2 == 0 ? "\n" : " ");
			}
		}

		void print_f64_registers()
		{
			for (size_t i = rv_freg_f0; i < P::freg_count; i++) {
				printf("%-4s: d %16.5f%s", rv_freg_name_sym[i],
					P::freg[i].r.d.val, (i + 1) % 2 == 0 ? "\n" : " ");
			}
		}

		template <typename D, typename R, typename V>
		void set_csr(D &dec, int mode, int op, int csr, R &reg, V value,
			const R write_mask = -1, const R read_mask = -1, const size_t shift = 0)
		{
			const int csr_mode = (csr >> 8) & 3, readonly = (csr >> 12) & 1;
			if (dec.rd != rv_ireg_x0) {
				P::ireg[dec.rd] = (mode >= csr_mode) ? (reg >> shift) & read_mask : 0;
			}
			if (readonly) return;
			switch (op) {
				case csr_rw: reg = (reg & ~(write_mask << shift)) | ((value & write_mask) << shift); break;
				case csr_rs: if (value) reg |= ((value & write_mask) << shift); break;
				case csr_rc: if (value) reg &= ~((value & write_mask) << shift); break;
			}
		}

		template <typename D, typename R, typename V>
		void get_csr(D &dec, int mode, int op, int csr, R &reg, V value)
		{
			const int csr_mode = (csr >> 8) & 3;
			if (dec.rd != rv_ireg_x0) {
				P::ireg[dec.rd] = (mode >= csr_mode) ? reg : 0;
			}
		}

		template <typename D, typename R, typename V>
		void set_csr_hi(D &dec, int mode, int op, int csr, R &reg, V value)
		{
			const int csr_mode = (csr >> 8) & 3, readonly = (csr >> 12) & 1;
			if (dec.rd != rv_ireg_x0) {
				P::ireg[dec.rd] = (mode >= csr_mode) ? s32(u32(reg >> 32)) : 0;
			}
			if (readonly) return;
			switch (op) {
				case csr_rw: reg = (u64(value) << 32) | (reg & 0xffffffffU); break;
				case csr_rs: if (value) reg |= (u64(value) << 32); break;
				case csr_rc: if (value) reg &= ~(u64(value) << 32) | 0xffffffffU; break;
			}
		}

		template <typename D, typename R, typename V>
		void get_csr_hi(D &dec, int mode, int op, int csr, R &reg, V value)
		{
			const int csr_mode = (csr >> 8) & 3;
			if (dec.rd != rv_ireg_x0) {
				P::ireg[dec.rd] = (mode >= csr_mode) ? s32(u32(reg >> 32)) : 0;
			}
		}
	};

}

#endif
