#Load_DRREP.jl
#################################################################################### SENSORS #################

module mymod
using LightXML
#import DataFrames # Even if using was used here full qualification of the functions in Base.Test and DataFrames is needed
#import Base.Test

type Sensor
	s_type::ASCIIString
	parameters::Array{Any,1}
	tot_columns::Integer
	ivl::Integer
	ovl::Integer
	columns::Array{Tuple{Int64,Int64,Int64},1}
	preprocessor::Function
	preprocessor_parameters::Any
	postprocessor::Function
	postprocessor_parameters::Any
	column_list::Any
	hres::Int64
	name::Function
	
	function Sensor(;N_Inputs::Integer = 1, Name=default_sensor, PreProcessor=default_preprocessor, PostProcessor=default_postprocessor, ColumnList=1:1, HRes=1)
		this = new()
		this.ivl = N_Inputs
		this.ovl = N_Inputs
		this.preprocessor = PreProcessor
		this.preprocessor_parameters = []
		this.postprocessor = PostProcessor
		this.postprocessor_parameters = []
		this.column_list = ColumnList
		this.name = Name
		this.parameters = []
		this.s_type = "std"
        this.tot_columns = -1
		this.columns = Array(Tuple{Int64,Int64,Int64},0)
        this.column_list = ColumnList
		this.hres = HRes
		this
	end
end

#Sensors take a partof the time series or dataset specified, or defaulted, and then compose a return vector.
function default_sensor(DataSet,CurPos,ColumnList,HRes)
    return DataSet[CurPos,ColumnList]'
end

function default_preprocessor(Input,Parameters)
    return Input
end

function default_postprocessor(Input,Parameters)
    return Input
end

function get_SlidingWindow(DataSet,CurPos,ColumnList,HRes)
    #A sliding window of protein properties of size HRes, including the protein at the current position.
    DefaultList = DataSet[CurPos-HRes+1:CurPos,ColumnList]
    return reshape(DefaultList,1,length(ColumnList)*length(CurPos-HRes+1:CurPos))'
end
#################################################################################### ACTUATORS #################
type Actuator
	a_type::ASCIIString
	parameters::Array{Any,1}
	tot_columns::Integer
	ivl::Integer
	ovl::Integer
	columns::Array{Tuple{Int64,Int64,Int64},1}
	preprocessor::Function
	preprocessor_parameters::Any
	postprocessor::Function
	postprocessor_parameters::Any
	column_list::Any
	name::Function
	
	function Actuator(;N_Inputs::Integer=1, Name = default_actuator,PreProcessor=default_preprocessor,PostProcessor=default_postprocessor,ColumnList=1:1)
		this = new()
		this.ivl = N_Inputs
		this.ovl = N_Inputs
		this.preprocessor = PreProcessor
		this.preprocessor_parameters = []
		this.postprocessor = PostProcessor
		this.postprocessor_parameters = []
		this.column_list = ColumnList
		this.name = Name
		this.parameters = []
        this.a_type = "std"
        this.tot_columns = -1
		this.columns = Array(Tuple{Int64,Int64,Int64},0)
        this.column_list = ColumnList
		this
	end
end

function default_actuator(Output,Expected,FitnessType)
    return Output
end

#################################################################################### NEURON #################
type NEURON
    weight_vector::Vector{Float64}
    advanced_weights
    bias::Float64
    agrf::Function
    af::Function
    neuron_type
    plasticity::Function
    parameters
    coordinates
    from
    to
    last_input::Vector{Float64}
	function NEURON(n_inputs::Integer;AF=get_af(), AGRF=get_agrf(),Parameters=[],InitWeights=[],InitBias=Inf)
		this = new()
		this.parameters = Parameters
		this.advanced_weights=[]
		this.neuron_type="std"
		this.plasticity=none
		this.coordinates=(rand(),rand())
		this.from=[]
		this.to=[]
		if InitWeights == []
			if AF == ngram_kernel
				N = rand(1:n_inputs)
				Alphabet = [65,82,78,68,67,69,81,71,72,73,76,75,77,70,80,83,84,87,89,86]
				Feature_Vector = [Alphabet[rand(1:length(Alphabet))] for i in 1:N]''
				Feature = (Feature_Vector[:,rand(1:size(Feature_Vector)[2])])
				this.weight_vector = Feature
				this.af = AF
				this.agrf = none
			elseif AF == mismatch_kernel
				N = rand(1:n_inputs)
				Alphabet = [65,82,78,68,67,69,81,71,72,73,76,75,77,70,80,83,84,87,89,86]
				Feature_Vector = [Alphabet[rand(1:length(Alphabet))] for i in 1:N]''
				Feature = (Feature_Vector[:,rand(1:size(Feature_Vector)[2])])
				this.weight_vector = Feature
				this.af = AF
				this.agrf = none
			else
			    this.weight_vector = rand(n_inputs) * 2 - 1
				this.bias = rand()*2-1
				this.agrf = AGRF
				this.af = AF
			end
		else
			this.weight_vector = InitWeights
			this.advanced_weights = []
			this.bias = InitBias
			this.af = AF
			this.agrf = AGRF
		end
		this.last_input=this.weight_vector
	    return this
	end
end


    function get_af(List)
        return List[rand(1:length(List))]
    end
	function get_af() 
		return af_list()[rand(1:length(af_list()))]
	end

	function af_list()
		return [ngram_kernel]#[cos,sin,sgn,sigmoid,bin]

	end

    function get_agrf(List)
        return List[rand(1:length(List))]
    end
	function get_agrf() 
		return agrf_list()[rand(1:length(agrf_list()))]
	end

	function agrf_list()
		return [af_dot]
	end

	function create_permutations(Alphabet,N)
		TotNGrams = length(Alphabet)^N
		InitNGram_List = fill(Alphabet[1],TotNGrams,N)
		Word = fill(Alphabet[1],N)
		create_permutations(InitNGram_List,Alphabet,N,Word,1,1)
		return InitNGram_List
	end

	function create_permutations(NGram_List,Alphabet,N,Word,Char_Index,Word_Index)
		for char in Alphabet
			Word[Char_Index] = char
			if Char_Index == N
				NGram_List[Word_Index,:] = Word
				Word_Index += 1
			else
				Word_Index = create_permutations(NGram_List,Alphabet,N,Word,Char_Index+1,Word_Index)
			end
		end
		return Word_Index
	end

#################################################################################### DRREP #################
type DeepRidgeRegressedPredictor
	neural_substrate::Matrix{NEURON}
	hidden_layer::Vector{NEURON}
	output_weights::Matrix{Float64}
	c::Integer
	initialized::Integer
	min_block_size::Integer
	pref_block_size::Integer
	learning_type::ASCIIString
	drrp_type::ASCIIString
	m::Matrix{Float64}
	sensors::Vector{Sensor}
	actuators::Vector{Actuator}
	normalizer::Function
	function DeepRidgeRegressedPredictor(n_inputs::Integer,n_hidden_neurons::Integer,n_classes::Integer, Sensors::Vector{Sensor}, Actuators::Vector{Actuator}; C = 10^2, LT="drrp", ET="std", Normalizer=proportionalize!,AFs=[sigmoid],AGRFs=[af_dot])#LT = "os_drrp")
		this = new()
        this.neural_substrate = Array(NEURON,0,0)
        this.m = Array(Float64,0,0)
		this.c = C
		this.min_block_size = n_hidden_neurons
		this.pref_block_size = n_hidden_neurons
		this.learning_type = LT
		this.drrp_type = ET
		this.initialized = 0
		this.sensors = Sensors
		this.actuators = Actuators
		this.normalizer = Normalizer
		(TotHiddenNodes,Neural_Layer) = create_layer(n_inputs,n_hidden_neurons,ET,AFs,AGRFs)

		this.hidden_layer = Neural_Layer
        this.output_weights = Array(Float64,n_classes, TotHiddenNodes)
		this
	end
end

function create_layer(IVL, Tot_Neurons,DRRP_Type,AFs,AGRFs)
		if DRRP_Type == "std"
			return (Tot_Neurons,[NEURON(IVL;AF=get_af(AFs),AGRF=get_agrf(AGRFs)) for i in 1:Tot_Neurons])
		end
	end

type INIT_CONFIG
    fitness_f::Function
    val_f::Function
    neuron_addition_attempts::Int
    tot_reset_attempts::Int
    add_neuron_range::Any
    activation_fs::Vector{Function}
    aggregation_fs::Vector{Function}
    c::Int
    ivl::Int
    ovl::Int
    brain_type::ASCIIString
    function INIT_CONFIG(;FitnessF=none, ValF=none, NeuronAdditionAttempts=20, TotResetAttempts=10, ANR=1:200, AFs=[sigmoid], AGRFs=[af_dot], C=10^2, IVL=12, OVL=1, Brain_Type="drrp")
        this = new()
        this.fitness_f = FitnessF
        this.val_f = ValF
        this.neuron_addition_attempts = NeuronAdditionAttempts
        this.tot_reset_attempts = TotResetAttempts
        this.add_neuron_range = ANR
        this.activation_fs = AFs
        this.aggregation_fs = AGRFs
        this.c = C
        this.ivl = IVL
        this.brain_type = Brain_Type
        return this
    end
end

type AGENT
	sensors::Vector{Sensor}
	actuators::Vector{Actuator}
	brain::Any
	spine::Any
	exoself::Any
	brain_type::ASCIIString
	ivl::Int
	ovl::Int
	id::Tuple{Float64,Float64}
	fitness::Float64
	init_config::INIT_CONFIG
	function AGENT(IVL::Integer, LayerSummary, TotHiddenNeurons, OVL::Integer, Sensors, Actuators, BrainType, IC)
		this = new()
		this.brain_type = BrainType
		if BrainType == "drrp"
			Brain = DeepRidgeRegressedPredictor(IVL,TotHiddenNeurons,OVL,Sensors,Actuators;ET="std",AFs=IC.activation_fs,AGRFs=IC.aggregation_fs)
		end
		this.brain = Brain
		this.sensors = Sensors
		this.actuators = Actuators
		this.ivl = IVL
		this.ovl = OVL
		this.id = (rand()+time(),rand()+time())
		this.spine = "void"
		this.exoself = "void"
		this.fitness = -1.0
		this.init_config = IC
		return this
	end
end

type COMMITTEE
    agent_list::Vector{AGENT}
    voting_method::Function
    id::Tuple{Float64,Float64}
    function COMMITTEE(;Voting_Method=scaled_sum)
        this = new()
        this.agent_list = []
        this.voting_method = Voting_Method
        this.id = (rand(),rand())
        return this
    end
end

function convert_Seq2Win(Input_URL)
    FileLink = open(Input_URL)
    Lines = readlines(FileLink)
    for SW_Length in [24]
        InputFile = open(string("./Input_Files/Converted_In.csv"),"w")
        Start_i = 1
        for Line in Lines
            Data_In = [Int64(Val[1]) for Val in Line]
            for SW_Pos in 1:(size(Data_In)[1] - SW_Length)
                Input = Data_In[SW_Pos:SW_Pos+SW_Length-1]
                Input = [Int64(Val[1]) for Val in Input]
                writecsv(InputFile,Input')
            end
            writecsv(InputFile,-1)
        end
        close(InputFile)
    end
end

#function main(ARGS...)
function main()
    println(join(ARGS,",")) 
    
    AgentId_FileNames = ["./Input_Files/SubDRREP_Id_List_24_0.7939644970414202_Time_1.458289308513196e9.csv"]
    println("Starting")
    
    if  length(ARGS) == 3
        Input_URL = ARGS[1]
        Output_URL = ARGS[2]
        StdDev_Multiplier = parse(ARGS[3])
        println("Input URL: $(ARGS[1])")
        println("Output URL: $(ARGS[2])")
        println("StdDev_Multiplier: $(ARGS[3])")
        convert_Seq2Win(Input_URL)
        Data_In = readcsv(string("./Input_Files/Converted_In.csv"))
        FileLink = open(Input_URL)
        Lines = readlines(FileLink)
        run_epitope_sequences(AgentId_FileNames,Data_In,Lines,Output_URL,StdDev_Multiplier)
    elseif length(ARGS) == 2
        Input_URL = ARGS[1]
        Output_URL = ARGS[2]
        StdDev_Multiplier = 0.0
        println("Input URL: $(ARGS[1])")
        println("Output URL: $(ARGS[2])")
        convert_Seq2Win(Input_URL)
        Data_In = readcsv(string("./Input_Files/Converted_In.csv"))
        FileLink = open(Input_URL)
        Lines = readlines(FileLink)
        run_epitope_sequences(AgentId_FileNames,Data_In,Lines,Output_URL,StdDev_Multiplier)

    else
        println("Please ues the following format:")
        println("julia [InputFilePath] [OutputFilePath] [StdDev_Multiplier]")
        println("If the third parameter is not used, a default value of 0.0 will be used instead")
    end
end

function run_epitope_sequences(AgentId_FileNames,Data_In,Orig_In_Lines,Output_URL,StdDev_Multiplier)
    Expected_URL = -1
    Result_Acc = []
    SeqRunP_Acc = []
    CurTime = time()
    RescaledVoteBased_PredictionsP = []
    SubSeqLocs=[]
    
    for i in 1:size(Data_In)[1]
        if Data_In[i,1] == -1
            SubSeqLocs = [SubSeqLocs;(i+(17*i+i))]
        end
    end
    if Expected_URL != -1
        Graph_File = open(string("./Input_Files/Committee_",SequenceFileName,"_Graphs_",CurTime,".csv"),"w")
        AUC_File = open(string("./Input_Files/Committee_",SequenceFileName,"_AUC_",CurTime,".csv"),"w")
        for i in 1:size(Data_Exp)[1]
            if Data_Exp[i,1] == -1
                SubSeqLocs = [SubSeqLocs;i]
            end
        end
    end
    File = open(string(Output_URL),"w")
    
    AgentIds_Acc = []
    for FileName in AgentId_FileNames
        Agent_Ids = readcsv(FileName)
        SW_Length = 24
        AgentIndex = 1
        TotAgents = length(Agent_Ids)
        for Agent_Id in Agent_Ids
            println("Loading Sub-DRREP:$Agent_Id.")
            Agent = read_XML_Agent(string("./Sub_DRREPs/SubDRREP_",Agent_Id,".xml"))
            (SeqPred,SeqPred_AUC) = agent_precut_sequence_run(Agent,Data_In,AgentIndex,TotAgents)
            SeqRunP_Acc = [SeqRunP_Acc; (SeqPred,SeqPred_AUC,Agent_Id)]
            AgentIndex = AgentIndex + 1
        end
    end
    PrevLoc = 1
    for k in 1:length(SubSeqLocs)
        Loc = SubSeqLocs[k]
        Input_Acc = []
        for Member in SeqRunP_Acc
            (SeqPred,SeqPred_AUC,Agent_Id) = Member
            InputP = SeqPred[k]
            Input_Acc = [Input_Acc; InputP]
        end
        (VoteBased_Predictions, RescaledVoteBased_Predictions) = committee_ScaledSummedOutputs(Input_Acc)
        PrevLoc = Loc+1
        
        if Expected_URL == -1
            RescaledThreshMean = mean(RescaledVoteBased_Predictions)
            Variance= sum([(Val-RescaledThreshMean)^2 for Val in RescaledVoteBased_Predictions])/size(RescaledVoteBased_Predictions)[1]
            StdDiv = sqrt(Variance)
#            BotThresh05 = RescaledThreshMean-0.5*StdDiv
#            TopThresh05 = RescaledThreshMean+0.5*StdDiv
#            BotThresh10 = RescaledThreshMean-1.0*StdDiv
#            TopThresh10 = RescaledThreshMean+1.0*StdDiv
#            BotThresh15 = RescaledThreshMean-1.5*StdDiv
#            TopThresh15 = RescaledThreshMean+1.5*StdDiv
            TopThreshCustom = RescaledThreshMean + StdDev_Multiplier*StdDiv
            BotThreshCustom = RescaledThreshMean - StdDev_Multiplier*StdDiv
            
            println(File,"DRREP Legend:")
            println(File,"1=amino acid position")
            println(File,"2=Amino acid Sequence")
            println(File,"3=Epitope prediction based on MeanScoreThreshold ($RescaledThreshMean) + StdDev_Multiplier*StdDev ($(StdDev_Multiplier*StdDiv))\n")
            println(File,"Threshold used: $(RescaledThreshMean + StdDev_Multiplier * StdDiv)")

            Tot_Residues = length(RescaledVoteBased_Predictions)
            LineSplits = 1:10:Tot_Residues
            Line = Orig_In_Lines[k]
            for i in 1:50:Tot_Residues
                for j in i:10:(i+49)
                    print(File,"$j         ")
                end
                print(File,"\n")


                for j in i:10:(i+49)
                    for Ind in j:min(j+9,Tot_Residues)
                        print(File,"$(Line[Ind])")
                    end
                end
                print(File,"\n")
                
                for j in i:10:(i+49)
                    for Ind in j:min(j+9,Tot_Residues)
                        Val = RescaledVoteBased_Predictions[Ind]
                        if Val > TopThreshCustom
                            print(File,"E")
                        else
                            print(File,".")
                        end
                     end
                end
                print(File,"\n")

            end
            print(File,"\n")
            
            EpitopeList = []
            EpiStart = -1
            Scores = []
            for i in 1:length(RescaledVoteBased_Predictions)
                if RescaledVoteBased_Predictions[i] > TopThreshCustom
                    if EpiStart == -1
                        EpiStart = i
                    end
                    Scores = [Scores; RescaledVoteBased_Predictions[i]]
                else
                    if (EpiStart != -1) || ((i == length(RescaledVoteBased_Predictions)) && (EpiStart != -1))
                        EpitopeList = [EpitopeList; (mean(Scores),Scores,EpiStart)]
                        EpiStart = -1
                        Scores = []
                    end
                end
            end
            if Scores != []
                EpitopeList = [EpitopeList; (mean(Scores),Scores,EpiStart)]
                EpiStart = -1
                Scores = []
            end
            EpitopeList = reverse(sort(EpitopeList))
            
            println(File,"Rank  Sequence    Start position  Score")
            for i in 1:length(EpitopeList)
                (MeanScore,Scores,EpiStart) = EpitopeList[i]
                println(File,"$i    $(Line[EpiStart:(EpiStart+length(Scores)-1)])    $EpiStart   $MeanScore")
            end
            print(File,"\n\n\n")
            
        else            
            Expected = Data_Exp[PrevLoc:Loc-1]
            VoteBased_AvgErr = avgerr(Expected,VoteBased_Predictions,Expected,[])
            VoteBased_RMS = rms(Expected,VoteBased_Predictions,Expected,[])
            VoteBased_Accuracy = accuracy(VoteBased_Predictions'',Expected)
            VoteBased_Accuracy_Spec75 = specificity_75(Expected,VoteBased_Predictions')
            VoteBased_AUC = auc([], VoteBased_Predictions'', Expected, [])
           
            RescaledVoteBased_AvgErr = avgerr([],RescaledVoteBased_Predictions,Expected,[])
            RescaledVoteBased_RMS = rms([],RescaledVoteBased_Predictions,Expected,[])
            RescaledVoteBased_Accuracy = accuracy(RescaledVoteBased_Predictions'',Expected)
            RescaledVoteBased_Accuracy_Spec75 = specificity_75(Expected,RescaledVoteBased_Predictions')
            RescaledVoteBased_AUC = auc([], RescaledVoteBased_Predictions'', Expected, [])
            
            PredMax = maximum(VoteBased_Predictions)
            PredMin = minimum(VoteBased_Predictions)
            Thresh75 = VoteBased_Accuracy_Spec75[4]
            RescaledPredMax = maximum(RescaledVoteBased_Predictions)
            RescaledPredMin = minimum(RescaledVoteBased_Predictions)
            RescaledThresh75 = RescaledVoteBased_Accuracy_Spec75[4]     

            Predictions_Bin75 = [VoteBased_Predictions[k] > Thresh75 ? 1:0 for k in 1:size(VoteBased_Predictions)[1]]
            RescaledPredictions_Bin75 = [RescaledVoteBased_Predictions[k] > RescaledThresh75 ? 1:0 for k in 1:size(RescaledVoteBased_Predictions)[1]]
            RescaledThreshMean = mean(RescaledVoteBased_Predictions)
            
            (TP_Mean,FP_Mean,TN_Mean,FN_Mean,NS_Mean) = calculateBinDZ_TP_FP_TN_FN(RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean,RescaledThreshMean)
            Accuracy_Mean = (TP_Mean+TN_Mean)/(TP_Mean+FP_Mean+TN_Mean+FN_Mean)
            Sensitivity_Mean = TP_Mean/(TP_Mean+FN_Mean)
            Specificity_Mean = TN_Mean/(TN_Mean+FP_Mean)
            CC_Mean = ((TP_Mean*TN_Mean)-(FP_Mean*FN_Mean))/sqrt((TN_Mean+FN_Mean)*(TN_Mean+FP_Mean)*(TP_Mean+FN_Mean)*(TP_Mean+FP_Mean))
            
            Variance= sum([(Val-RescaledThreshMean)^2 for Val in RescaledVoteBased_Predictions])/size(RescaledVoteBased_Predictions)[1]
            StdDiv = sqrt(Variance)

            println(File,"RescaledPredictions:: Mean:$(RescaledThreshMean), Variance:$(Variance), StdDiv:$(StdDiv), Max:$(maximum(RescaledVoteBased_Predictions)), Min:$(minimum(RescaledVoteBased_Predictions))")
            println(File,"RescaledVote Mean:: Accuracy:$(Accuracy_Mean), Sensitivity:$(Sensitivity_Mean), Specificity:$(Specificity_Mean), CC:($CC_Mean), ThreshMean:$RescaledThreshMean, TP:$(TP_Mean) FP:$(FP_Mean) TN:$(TN_Mean) FN:$(FN_Mean)")

            RescaledVoteBased_PredictionsP = [RescaledVoteBased_PredictionsP;(RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean,StdDiv)]          
            (TP_DZ1,FP_DZ1,TN_DZ1,FN_DZ1,NS_DZ1) = calculateBinDZ_TP_FP_TN_FN(RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean-0.5*StdDiv,RescaledThreshMean+0.5*StdDiv)
            Accuracy_DZ1 = (TP_DZ1+TN_DZ1)/(TP_DZ1+FP_DZ1+TN_DZ1+FN_DZ1)
            Sensitivity_DZ1 = TP_DZ1/(TP_DZ1+FN_DZ1)
            Specificity_DZ1 = TN_DZ1/(TN_DZ1+FP_DZ1)
            CC_DZ1 = ((TP_DZ1*TN_DZ1)-(FP_DZ1*FN_DZ1))/sqrt((TN_DZ1+FN_DZ1)*(TN_DZ1+FP_DZ1)*(TP_DZ1+FN_DZ1)*(TP_DZ1+FP_DZ1))
            println(File,"RescaledVote DZ +/-0.5*StdDiv::   Accuracy:$(Accuracy_DZ1), Sensitivity:$(Sensitivity_DZ1), Specificity:$(Specificity_DZ1), CC:($CC_DZ1), TP:$(TP_DZ1) FP:$(FP_DZ1) TN:$(TN_DZ1) FN:$(FN_DZ1) NS_DZ:$(NS_DZ1)")
            (TP_DZ2,FP_DZ2,TN_DZ2,FN_DZ2,NS_DZ2) = calculateBinDZ_TP_FP_TN_FN(RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean-1.0*StdDiv,RescaledThreshMean+1.0*StdDiv)
            Accuracy_DZ2 = (TP_DZ2+TN_DZ2)/(TP_DZ2+FP_DZ2+TN_DZ2+FN_DZ2)
            Sensitivity_DZ2 = TP_DZ2/(TP_DZ2+FN_DZ2)
            Specificity_DZ2 = TN_DZ2/(TN_DZ2+FP_DZ2)
            CC_DZ2 = ((TP_DZ2*TN_DZ2)-(FP_DZ2*FN_DZ2))/sqrt((TN_DZ2+FN_DZ2)*(TN_DZ2+FP_DZ2)*(TP_DZ2+FN_DZ2)*(TP_DZ2+FP_DZ2))
            println(File,"RescaledVote DZ +/-1.0*StdDiv::   Accuracy:$(Accuracy_DZ2), Sensitivity:$(Sensitivity_DZ2), Specificity:$(Specificity_DZ2), CC:($CC_DZ2), TP:$(TP_DZ2) FP:$(FP_DZ2) TN:$(TN_DZ2) FN:$(FN_DZ2) NS_DZ:$(NS_DZ2)")
            (TP_DZ3,FP_DZ3,TN_DZ3,FN_DZ3,NS_DZ3) = calculateBinDZ_TP_FP_TN_FN(RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean-1.5*StdDiv,RescaledThreshMean+1.5*StdDiv)
            Accuracy_DZ3 = (TP_DZ3+TN_DZ3)/(TP_DZ3+FP_DZ3+TN_DZ3+FN_DZ3)
            Sensitivity_DZ3 = TP_DZ3/(TP_DZ3+FN_DZ3)
            Specificity_DZ3 = TN_DZ3/(TN_DZ3+FP_DZ3)
            CC_DZ3 = ((TP_DZ3*TN_DZ3)-(FP_DZ3*FN_DZ3))/sqrt((TN_DZ3+FN_DZ3)*(TN_DZ3+FP_DZ3)*(TP_DZ3+FN_DZ3)*(TP_DZ3+FP_DZ3))
            println(File,"RescaledVote DZ +/-1.5*StdDiv::   Accuracy:$(Accuracy_DZ3), Sensitivity:$(Sensitivity_DZ3), Specificity:$(Specificity_DZ3), CC:($CC_DZ3), TP:$(TP_DZ3) FP:$(FP_DZ3) TN:$(TN_DZ3) FN:$(FN_DZ3) NS_DZ:$(NS_DZ3)")

            println(File,"EqualVote Spec75%::    Accuracy:$(VoteBased_Accuracy_Spec75[2]), Thresh75:$Thresh75, TP:$(VoteBased_Accuracy_Spec75[5]) FP:$(VoteBased_Accuracy_Spec75[6]) TN:$(VoteBased_Accuracy_Spec75[7]) FN:$(VoteBased_Accuracy_Spec75[8]), AUC:$(VoteBased_AUC)")
            println(File,"RescaledVote Spec75%:: Accuracy:$(RescaledVoteBased_Accuracy_Spec75[2]), Thresh75:$RescaledThresh75, TP:$(RescaledVoteBased_Accuracy_Spec75[5]) FP:$(RescaledVoteBased_Accuracy_Spec75[6]) TN:$(RescaledVoteBased_Accuracy_Spec75[7]) FN:$(RescaledVoteBased_Accuracy_Spec75[8]), AUC:$(RescaledVoteBased_AUC)")
            for k in 1:size(VoteBased_Predictions)[1]
                println(File,"$(Expected[k]), $(VoteBased_Predictions[k]), $(Predictions_Bin75[k]), $(RescaledVoteBased_Predictions[k]), $(RescaledPredictions_Bin75[k])")
            end
            println(File,"####################")
            Result_Acc = [Result_Acc; (VoteBased_Accuracy_Spec75, RescaledVoteBased_Accuracy_Spec75, VoteBased_AUC,RescaledVoteBased_AUC, TP_Mean, FP_Mean, TN_Mean, FN_Mean, RescaledThreshMean, Variance, StdDiv, TP_DZ1, FP_DZ1, TN_DZ1, FN_DZ1, NS_DZ1, TP_DZ2, FP_DZ2, TN_DZ2, FN_DZ2, NS_DZ2, TP_DZ3, FP_DZ3, TN_DZ3, FN_DZ3, NS_DZ3)]
        end
    end

    if Expected_URL != -1 
        graph_accuracy(Graph_File,RescaledVoteBased_PredictionsP)
        close(Graph_File)
        graph_auc(AUC_File,RescaledVoteBased_PredictionsP)
        close(AUC_File)
    end
    
    if Expected_URL == -1 
        close(File)
    else
        Mean_VoteBased_Accuracy_Spec75= mean([Result[1][2] for Result in Result_Acc]) 
        Mean_RescaledVoteBased_Accuracy_Spec75= mean([Result[2][2] for Result in Result_Acc])
        Mean_VoteBased_AUC= mean([Result[3] for Result in Result_Acc])
        Mean_RescaledVoteBased_AUC= mean([Result[4] for Result in Result_Acc])
        TP_M= sum([Result[5] for Result in Result_Acc])
        FP_M= sum([Result[6] for Result in Result_Acc])
        TN_M= sum([Result[7] for Result in Result_Acc])
        FN_M= sum([Result[8] for Result in Result_Acc])
        AccM = (TP_M+TN_M)/(TP_M+FP_M+TN_M+FN_M)
        SnsM = TP_M/(TP_M+FN_M)
        SpcM = TN_M/(TN_M+FP_M)
        CCM = ((TP_M*TN_M)-(FP_M*FN_M))/sqrt((TN_M+FN_M)*(TN_M+FP_M)*(TP_M+FN_M)*(TP_M+FP_M))
        TotResidues_M = TP_M+FP_M+TN_M+FN_M
        
        Mean_Threshold=mean([Result[9] for Result in Result_Acc])
        Mean_Variance=mean([Result[10] for Result in Result_Acc])
        Mean_StdDiv=mean([Result[11] for Result in Result_Acc])
        
        TP_DZ1= sum([Result[12] for Result in Result_Acc])
        FP_DZ1= sum([Result[13] for Result in Result_Acc])
        TN_DZ1= sum([Result[14] for Result in Result_Acc])
        FN_DZ1= sum([Result[15] for Result in Result_Acc])
        NS_DZ1= sum([Result[16] for Result in Result_Acc])
        Accuracy_DZ1 = (TP_DZ1+TN_DZ1)/(TP_DZ1+FP_DZ1+TN_DZ1+FN_DZ1)
        Sensitivity_DZ1 = TP_DZ1/(TP_DZ1+FN_DZ1)
        Specificity_DZ1 = TN_DZ1/(TN_DZ1+FP_DZ1)
        CC_DZ1 = ((TP_DZ1*TN_DZ1)-(FP_DZ1*FN_DZ1))/sqrt((TN_DZ1+FN_DZ1)*(TN_DZ1+FP_DZ1)*(TP_DZ1+FN_DZ1)*(TP_DZ1+FP_DZ1))
        TotResidues_DZ1 = TP_DZ1+FP_DZ1+TN_DZ1+FN_DZ1
        
        TP_DZ2= sum([Result[17] for Result in Result_Acc])
        FP_DZ2= sum([Result[18] for Result in Result_Acc])
        TN_DZ2= sum([Result[19] for Result in Result_Acc])
        FN_DZ2= sum([Result[20] for Result in Result_Acc])
        NS_DZ2= sum([Result[21] for Result in Result_Acc])
        Accuracy_DZ2 = (TP_DZ2+TN_DZ2)/(TP_DZ2+FP_DZ2+TN_DZ2+FN_DZ2)
        Sensitivity_DZ2 = TP_DZ2/(TP_DZ2+FN_DZ2)
        Specificity_DZ2 = TN_DZ2/(TN_DZ2+FP_DZ2)
        CC_DZ2 = ((TP_DZ2*TN_DZ2)-(FP_DZ2*FN_DZ2))/sqrt((TN_DZ2+FN_DZ2)*(TN_DZ2+FP_DZ2)*(TP_DZ2+FN_DZ2)*(TP_DZ2+FP_DZ2))
        TotResidues_DZ2 = TP_DZ2+FP_DZ2+TN_DZ2+FN_DZ2
        
        TP_DZ3= sum([Result[22] for Result in Result_Acc])
        FP_DZ3= sum([Result[23] for Result in Result_Acc])
        TN_DZ3= sum([Result[24] for Result in Result_Acc])
        FN_DZ3= sum([Result[25] for Result in Result_Acc])
        NS_DZ3= sum([Result[26] for Result in Result_Acc])
        Accuracy_DZ3 = (TP_DZ3+TN_DZ3)/(TP_DZ3+FP_DZ3+TN_DZ3+FN_DZ3)
        Sensitivity_DZ3 = TP_DZ3/(TP_DZ3+FN_DZ3)
        Specificity_DZ3 = TN_DZ3/(TN_DZ3+FP_DZ3)
        CC_DZ3 = ((TP_DZ3*TN_DZ3)-(FP_DZ3*FN_DZ3))/sqrt((TN_DZ3+FN_DZ3)*(TN_DZ3+FP_DZ3)*(TP_DZ3+FN_DZ3)*(TP_DZ3+FP_DZ3))
        TotResidues_DZ3 = TP_DZ3+FP_DZ3+TN_DZ3+FN_DZ3
          
        println(File,"SequenceFileName Committee Mean parameters:: Threshold:$(Mean_Threshold), Variance:$(Mean_Variance), StdDiv:$(Mean_StdDiv)")
        println(File,"$SequenceFileName Committee Rescaled Mean:: Accuracy:$(AccM), Sensitivity:$(SnsM), Specificity:$(SpcM), CC:$(CCM), TP:$(TP_M), FP:$(FP_M), TN:$(TN_M), FN:$(FN_M), TotResidues:$(TotResidues_M)")
        println(File,"$SequenceFileName Committee Rescaled DZ1:: Accuracy:$(Accuracy_DZ1), Sensitivity:$(Sensitivity_DZ1), Specificity:$(Specificity_DZ1), CC:$(CC_DZ1), TP:$(TP_DZ1), FP:$(FP_DZ1), TN:$(TN_DZ1), FN:$(FN_DZ1), NS:$(NS_DZ1), TotRes:$(TotResidues_DZ1)")
        println(File,"$SequenceFileName Committee Rescaled DZ2:: Accuracy:$(Accuracy_DZ2), Sensitivity:$(Sensitivity_DZ2), Specificity:$(Specificity_DZ2), CC:$(CC_DZ2), TP:$(TP_DZ2), FP:$(FP_DZ2), TN:$(TN_DZ2), FN:$(FN_DZ2), NS:$(NS_DZ2), TotRes:$(TotResidues_DZ2)")
        println(File,"$SequenceFileName Committee Rescaled DZ3:: Accuracy:$(Accuracy_DZ3), Sensitivity:$(Sensitivity_DZ3), Specificity:$(Specificity_DZ3), CC:$(CC_DZ3), TP:$(TP_DZ3), FP:$(FP_DZ3), TN:$(TN_DZ3), FN:$(FN_DZ3), NS:$(NS_DZ3), TotRes:$(TotResidues_DZ3)")
        println(File,"$SequenceFileName Committee Spec75%:: Accuracy:$(Mean_VoteBased_Accuracy_Spec75), Rescaled Accuracy:$(Mean_RescaledVoteBased_Accuracy_Spec75), AUC:$(Mean_VoteBased_AUC), RescaledAUC:$(Mean_RescaledVoteBased_AUC)")
        close(File)
    end
    #return (Mean_VoteBased_Accuracy_Spec75,Mean_RescaledVoteBased_Accuracy_Spec75,Mean_VoteBased_AUC,Mean_RescaledVoteBased_AUC)
end

    function agent_precut_sequence_run(Agent,ArrayIn,AgentIndex,TotAgents)
        SeqPred_Acc = []
        PredAUC_Acc = 0
        In_i = 1
        Exp_i = 1
        Exp_i_Start = 1
        TotSubSeq=0
        SW_Length = Agent.ivl
        print("Postprocessing sliding window outputs for SubDrep-$AgentIndex of $TotAgents:")
        for i in 1:size(ArrayIn)[1]
            if (ArrayIn[i,1] == -1)
                gc()
                Data_In = ArrayIn[In_i:i-1,:]''
                In_i = i+1
                OutputArray = agent_predict(Agent,Data_In)'
                SeqRun_Prediction = postproc_SlidingWindowOutputs(OutputArray'',SW_Length)
                SeqPred_Acc = [SeqPred_Acc; (SeqRun_Prediction,Float64(Agent.fitness))]
                PredAUC_Acc += Float64(Agent.fitness)
                TotSubSeq += 1
            end
        end
        println("Done.")
        AUC = PredAUC_Acc/TotSubSeq
        return (SeqPred_Acc,AUC)
    end
    
        function postproc_SlidingWindowOutputs(OutputArray,SW_Length)#Single full output slice of a single continues sequence.
            VoteBased_Predictions = 0.0
            VoteBased_TstAccuracy = 0.0
            VoteBased_TstAUC = 0.0
            VoteBased_Pred_Size = (size(OutputArray)[1]-1+SW_Length)
            print(".")
            VoteBased_Pred_Exp = zeros(Float64,VoteBased_Pred_Size)''
            ###vvvNORMALIZATION OF THE SUMMED SUB WINDOWS
            for i in 1:size(OutputArray)[1]
                Val = OutputArray[i]
                for k in i:(i+SW_Length-1)
                    VoteBased_Pred_Exp[k] = VoteBased_Pred_Exp[k] + Val
                end
                
                if i < SW_Length
                    VoteBased_Pred_Exp[i] = VoteBased_Pred_Exp[i]/i
                else
                    VoteBased_Pred_Exp[i] = VoteBased_Pred_Exp[i]/SW_Length
                end
            end
            for i in (size(OutputArray)[1]+1):size(VoteBased_Pred_Exp)[1]
                Val = size(OutputArray)[1]+SW_Length-i
                VoteBased_Pred_Exp[i] = VoteBased_Pred_Exp[i]/Val
            end
            ###^^^NORMALIZATION OF THE SUMMED SUB WINDOWS
            return VoteBased_Pred_Exp
        end

        function auc(Expected, Output, n=200)
            ThresholdList = minimum(Output):1/n:maximum(Output)
            TP_FP_TN_FN_List = [calculateBinDZ_TP_FP_TN_FN(Output, Expected,Threshold,Threshold) for Threshold in ThresholdList]
            Acc = []
            for TP_FP_TN_FN in TP_FP_TN_FN_List
                if (TP_FP_TN_FN[3]+TP_FP_TN_FN[2]) == 0
                    X = 1
                else
                    X = 1-TP_FP_TN_FN[3]/(TP_FP_TN_FN[3]+TP_FP_TN_FN[2])
                end
                if (TP_FP_TN_FN[1]+TP_FP_TN_FN[4]) == 0
                    Y = 0
                else
                    Y = TP_FP_TN_FN[1]/(TP_FP_TN_FN[1]+TP_FP_TN_FN[4])
                end
                Acc = [Acc; (X,Y)]
            end
            AUC_Graph = sort(Acc)
            Area = 0.0
            dx_step = 1
            for i in 2:length(AUC_Graph)
                dx = AUC_Graph[i][1] - AUC_Graph[i-dx_step][1]  #delta FPR
                dy = (AUC_Graph[i][2] - AUC_Graph[i-dx_step][2])/2  #delta TPR
                Area += dx*AUC_Graph[i-dx_step][2] + dx*dy  #0.5 * width * (height_(i) + height_(i-1))
            end
            if Area < 0.5
                return 1 - Area
            else
                return Area
            end
        end
        
        function auc(Val_TrnData, ValOut, ValExp, Sensors;Parameters=0)
            if size(ValOut)[2] == 1
                Class_Predictions = vec(ValOut)
                Class_Expected = round(Int64,vec(ValExp))
                if (sum(Class_Predictions) != 0) && (sum(Class_Predictions) != length(Class_Predictions))
                    AUC = auc(Class_Expected, Class_Predictions, 200)
                    return AUC
                else
                    return 0
                end
            else
                (Class_Predictions,Scores) = to_MaxClass(ValOut)
                Class_Expected = vec(sparse_bin_to_int(ValExp,size(ValExp)[2])) .- 1
                if (sum(Class_Predictions) != 0) && (sum(Class_Predictions) != length(Class_Predictions))
                    AUC = auc(Class_Expected, (Class_Predictions .- 1,Scores), 200)
                    return AUC
                else
                    return 0
                end
            end
        end
    function committee_ScaledSummedOutputs(Outputs;WeightList=[])
        VoteBased_Predictions = []
        RescaledVoteBased_Predictions = []
        AUC_Sum = sum([Outputs[i][2] for i in 1:size(Outputs)[1]])
        for SeqRun_Result in Outputs
            if VoteBased_Predictions == []
                VoteBased_Predictions = rescale(SeqRun_Result[1])
                RescaledVoteBased_Predictions = rescale(SeqRun_Result[1]) .* (SeqRun_Result[2]/AUC_Sum)
            else
                VoteBased_Predictions = VoteBased_Predictions .+ rescale(SeqRun_Result[1])
                RescaledVoteBased_Predictions = RescaledVoteBased_Predictions .+ (rescale(SeqRun_Result[1]) .* (SeqRun_Result[2]/AUC_Sum))
            end
        end
        return (VoteBased_Predictions,RescaledVoteBased_Predictions)
    end
    
    function avgerr(Val_TrnData,ValOut,ValExp,Sensors)#Calculates AvgErr, returns 1/AvgErr
        AvgErr = sum([abs(Val) for Val in (ValOut .- ValExp)])
        return 1/AvgErr
    end
    
    function rms(Val_TrnData,ValOut,ValExp,Sensors)#Calculates RMS, returns 1/RMS
        RMS = sqrt(sum([Val*Val for Val in (ValOut .- ValExp)]))
        return (1/RMS)
    end

    function accuracy(Output,Expected;TrimFlag=true,Threshold="mean")
        Mean = mean(Output)
        (TP,FP,TN,FN,NS) = calculateBinDZ_TP_FP_TN_FN(Output,Expected,Mean,Mean)
        Accuracy = (TP+TN)/size(Output)[1]
        return Accuracy
    end
    
        function calculateBinDZ_TP_FP_TN_FN(Output,Expected,Min_DZ,Max_DZ)
            TP=0
            FP=0
            TN=0
            FN=0
            NS=0
            for i in 1:size(Output)[1]
                if (Output[i] >= Max_DZ)
                    Out = 1 
                elseif (Output[i] <= Min_DZ)
                    Out = 0 
                else
                    Out = -1
                end
                
                if (Out==1) && (Expected[i]==1)
                    TP+=1
                elseif (Out==0) && (Expected[i]==0)
                    TN+=1
                elseif (Out==1) && (Expected[i]==0)
                    FP+=1
                elseif (Out==0) && (Expected[i]==1)
                    FN+=1
                elseif (Out == -1)
                    NS+=1
                end    
            end
            return (TP, FP, TN, FN, NS)
        end
        
    function specificity_75(Tst_Data,Pred)
        TstD = []
        TstE = Tst_Data[:,1]
        Target_Specificity = 0.75
        TotThresholds = 20
        Max = maximum(Pred)
        Min = minimum(Pred)
        Step = (Max-Min)/(TotThresholds-1)
        if Step == 0
            ThresholdList = [Min]
        else
            ThresholdList = Min:Step:Max
        end
        Best_Spec = (-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf)
        for Thresh in ThresholdList
            (TP,FP,TN,FN,NS) = calculateBinDZ_TP_FP_TN_FN(Pred',TstE'',Thresh,Thresh)
            Specificity = TN/(TN+FP)
            Accuracy = (TP+TN)/size(Pred')[1]
            CC = ((TP*TN)-(FP*FN))/sqrt((TN+FN)*(TN+FP)*(TP+FN)*(TP+FP))
            if 1/(Specificity-Target_Specificity) > Best_Spec[1]
                Best_Spec = (1/(Specificity-Target_Specificity), Specificity, Accuracy, CC, Thresh,TP,FP,TN,FN)
            end
        end    
        return (Best_Spec[2],Best_Spec[3],Best_Spec[4],Best_Spec[5], Best_Spec[6], Best_Spec[7], Best_Spec[8], Best_Spec[9])
    end

    function graph_accuracy(Graph_File,RescaledVoteBased_PredictionsP)
        write(Graph_File,"#Expected\n")
        g = 1
        for Val in RescaledVoteBased_PredictionsP
            (RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean,StdDiv) = Val
            TotRows = size(RescaledVoteBased_Predictions)[1]
            for i in 1:TotRows
                write(Graph_File,"$g ")
                write(Graph_File,"$(2*(Data_Exp[i] - 0.5))\n")
                g = g+1
            end
        end
        write(Graph_File,"\n\n")

        write(Graph_File,"#Predictions\n")
        g = 1
        for Val in RescaledVoteBased_PredictionsP
            (RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean,StdDiv) = Val
            TotRows = size(RescaledVoteBased_Predictions)[1]
            Out = 0
            for i in 1:TotRows
                write(Graph_File,"$g ")
                write(Graph_File,"$(RescaledVoteBased_Predictions[i])\n")
                g = g+1
            end
        end
        write(Graph_File,"\n\n")
            
        write(Graph_File,"#Predictions: +/- 0.5 StdDiv\n")
        g = 1
        for Val in RescaledVoteBased_PredictionsP
            (RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean,StdDiv) = Val
            TotRows = size(RescaledVoteBased_Predictions)[1]
            for i in 1:TotRows
                Val = RescaledVoteBased_Predictions[i]
                if Val > (RescaledThreshMean+0.5*StdDiv)
                    Out=0.25
                    write(Graph_File,"$g ")
                    write(Graph_File,"$(Out)\n")
                elseif Val < (RescaledThreshMean-0.5*StdDiv)
                    Out=-0.25
                    write(Graph_File,"$g ")
                    write(Graph_File,"$(Out)\n")
                end
                g = g+1
            end
        end
        write(Graph_File,"\n\n")
        
        write(Graph_File,"#Predictions: +/- 1.0 StdDiv\n")
        g = 1
        for Val in RescaledVoteBased_PredictionsP
            (RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean,StdDiv) = Val
            TotRows = size(RescaledVoteBased_Predictions)[1]
            for i in 1:TotRows
                Val = RescaledVoteBased_Predictions[i]
                if Val > (RescaledThreshMean+1*StdDiv)
                    Out=0.5
                    write(Graph_File,"$g ")
                    write(Graph_File,"$(Out)\n")
                elseif Val < (RescaledThreshMean-1*StdDiv)
                    Out=-0.5
                    write(Graph_File,"$g ")
                    write(Graph_File,"$(Out)\n")
                end
                g = g+1
            end
        end
        write(Graph_File,"\n\n")
        
        write(Graph_File,"#Predictions: +/- 1.5 StdDiv\n")
        g = 1
        for Val in RescaledVoteBased_PredictionsP
            (RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean,StdDiv) = Val
            TotRows = size(RescaledVoteBased_Predictions)[1]
            for i in 1:TotRows
                Val = RescaledVoteBased_Predictions[i]
                if Val > (RescaledThreshMean+1.5*StdDiv)
                    Out=0.75
                    write(Graph_File,"$g ")
                    write(Graph_File,"$(Out)\n")
                elseif Val < (RescaledThreshMean-1.5*StdDiv)
                    Out=-0.75
                    write(Graph_File,"$g ")
                    write(Graph_File,"$(Out)\n")
                end
                g = g+1
            end
        end
    end

    function graph_auc(AUC_File,RescaledVoteBased_PredictionsP)
        TotThresholds=100
        ThresholdIndexList = 0:TotThresholds
        TP_FP_TN_FN_Array = zeros(TotThresholds+1,4)
        write(AUC_File,"#TPR vs FPR\n")
        Index = 1
        for ThresholdIndex in ThresholdIndexList
            TP_Acc = 0
            FP_Acc = 0
            TN_Acc = 0
            FN_Acc = 0
            for Val in RescaledVoteBased_PredictionsP
                (RescaledVoteBased_Predictions,Data_Exp,RescaledThreshMean,StdDiv) = Val
                Max = maximum(RescaledVoteBased_Predictions)
                Min = minimum(RescaledVoteBased_Predictions)
                Range = Max-Min
                Threshold = Min + ThresholdIndex*(Range/TotThresholds)
                (TP,FP,TN,FN,NS) = calculateBinDZ_TP_FP_TN_FN(RescaledVoteBased_Predictions,Data_Exp,Threshold,Threshold)
                TP_Acc = TP_Acc + TP
                FP_Acc = FP_Acc + FP
                TN_Acc = TN_Acc + TN
                FN_Acc = FN_Acc + FN
            end
            TP_FP_TN_FN_Array[Index,:] = [TP_Acc FP_Acc TN_Acc FN_Acc]
            Index = Index+1
            Specificity = TN_Acc/(TN_Acc+FP_Acc)
            Sensitivity = TP_Acc/(TP_Acc+FN_Acc)
            write(AUC_File,"$(1-Specificity) $(Sensitivity)\n")
        end
        return TP_FP_TN_FN_Array
    end
    
    function read_XML_Agent(FileName::ASCIIString)
        Agent = AGENT(1,[1],0,1,[Sensor()],[Actuator()],"drrp",INIT_CONFIG())
        xdoc = LightXML.parse_file(FileName)
        xroot = LightXML.root(xdoc)
        read_XML_Agent_Struct!(Agent,xroot)
        return Agent
    end

    function read_XML_Agent_Struct!(Agent,xroot)
        xs_Id = LightXML.find_element(xroot, "Id")
        xs_Sensors = LightXML.find_element(xroot,"Sensors")
        xs_Actuators = LightXML.find_element(xroot,"Actuators")
        xs_BrainType = LightXML.find_element(xroot,"Brain_Type")
        xs_Brain = LightXML.find_element(xroot,"Brain")
        xs_Spine = LightXML.find_element(xroot,"Spine")
        xs_Exoself = LightXML.find_element(xroot,"Exoself")
        xs_IVL = LightXML.find_element(xroot,"IVL")
        xs_OVL = LightXML.find_element(xroot,"OVL")
        xs_Fitness = LightXML.find_element(xroot,"Fitness")
        xs_InitConfig = LightXML.find_element(xroot,"Init_Config")
        
        Agent.id = eval(parse(LightXML.content(xs_Id)))
        Agent.sensors = read_XML_Sensors(xs_Sensors)
        Agent.actuators = read_XML_Actuators(xs_Actuators)
        Agent.brain_type = LightXML.content(xs_BrainType)
        if Agent.brain_type == "drrp"
            Agent.brain = read_XML_DRRP(xs_Brain)
        end
        Agent.spine = LightXML.content(xs_Spine)
        Agent.exoself = LightXML.content(xs_Exoself)
        Agent.ivl = eval(parse(LightXML.content(xs_IVL)))
        Agent.ovl = eval(parse(LightXML.content(xs_OVL)))
        Agent.fitness = eval(parse(LightXML.content(xs_Fitness)))
    end

            
            function read_XML_InitConfig(xs_InitConfig)
                xs_FitnessF = LightXML.find_element(xs_InitConfig,"Fitness_Function")
                xs_ValF = LightXML.find_element(xs_InitConfig,"Validation_Function")
                xs_NeuronAdditionAttempts = LightXML.find_element(xs_InitConfig,"Neuron_Addition_Attempts")
                xs_TotResetAttempts = LightXML.find_element(xs_InitConfig,"Tot_Reset_Attempts")
                xs_AddNeuronRange = LightXML.find_element(xs_InitConfig,"Add_Neuron_Range")
                xs_ActivationFs = LightXML.find_element(xs_InitConfig,"Activation_Functions")
                xs_AggregationFs = LightXML.find_element(xs_InitConfig,"Aggregation_Functions")
                
                InitConfig = INIT_CONFIG()
                InitConfig.fitness_f = eval(parse(LightXML.content(xs_FitnessF)))
                InitConfig.val_f = eval(parse(LightXML.content(xs_ValF)))
                InitConfig.neuron_addition_attempts = eval(parse(LightXML.content(xs_NeuronAdditionAttempts)))
                InitConfig.tot_reset_attempts = eval(parse(LightXML.content(xs_TotResetAttempts)))
                InitConfig.add_neuron_range = eval(parse(LightXML.content(xs_AddNeuronRange)))
                InitConfig.activation_fs = eval(parse(LightXML.content(xs_ActivationFs)))
                InitConfig.aggregation_fs = eval(parse(LightXML.content(xs_AggregationFs)))
                return InitConfig
            end
                    
            function read_XML_Sensors(xs_Sensors)
                Sensors=[]
                for xs_Sensor in LightXML.child_elements(xs_Sensors)
                    Sensors = [read_XML_Sensor(xs_Sensor);Sensors]
                end
                return reverse(Sensors)
            end
            
            function read_XML_Sensor(xs_Sensor)
                xs_SensorType = LightXML.find_element(xs_Sensor, "S_Type")
                xs_Parameters = LightXML.find_element(xs_Sensor, "Parameters")
                xs_TotColumns = LightXML.find_element(xs_Sensor, "Tot_Columns")
                xs_IVL = LightXML.find_element(xs_Sensor,"IVL")
                xs_OVL = LightXML.find_element(xs_Sensor,"OVL")
                xs_Columns = LightXML.find_element(xs_Sensor,"Columns")
                xs_Preprocessor = LightXML.find_element(xs_Sensor,"Preprocessor")
                xs_PreprocessorParameters = LightXML.find_element(xs_Sensor,"Preprocessor_Parameters")
                xs_Postprocessor = LightXML.find_element(xs_Sensor,"Postprocessor")
                xs_PostprocessorParameters = LightXML.find_element(xs_Sensor,"Postprocessor_Parameters")
                xs_ColumnList = LightXML.find_element(xs_Sensor,"Column_List")
                xs_HRes = LightXML.find_element(xs_Sensor,"HRes")
                xs_Name = LightXML.find_element(xs_Sensor,"Name")
                
                S = Sensor()
                S.s_type = LightXML.content(xs_SensorType)
                S.parameters = eval(parse(LightXML.content(xs_Parameters)))
                S.tot_columns = eval(parse(LightXML.content(xs_TotColumns)))
                S.ivl = eval(parse(LightXML.content(xs_IVL)))
                S.ovl = eval(parse(LightXML.content(xs_OVL)))
                S.columns = eval(parse(LightXML.content(xs_Columns)))
                S.preprocessor = eval(parse(LightXML.content(xs_Preprocessor)))
                S.preprocessor_parameters = eval(parse(LightXML.content(xs_PreprocessorParameters)))
                S.postprocessor = eval(parse(LightXML.content(xs_Postprocessor)))
                S.postprocessor_parameters = eval(parse(LightXML.content(xs_PostprocessorParameters)))
                S.column_list = eval(parse(LightXML.content(xs_ColumnList)))
                S.hres = eval(parse(LightXML.content(xs_HRes)))
                S.name = eval(parse(LightXML.content(xs_Name)))
                return S
            end
                    
            function read_XML_Actuators(xs_Actuators)
                Actuators = []
                for xs_Actuator in LightXML.child_elements(xs_Actuators)
                    Actuators = [read_XML_Actuator(xs_Actuator);Actuators]
                end
                return reverse(Actuators)
            end
            
            function read_XML_Actuator(xs_Actuator)
                xs_ActuatorType = LightXML.find_element(xs_Actuator, "A_Type")
                xs_Parameters = LightXML.find_element(xs_Actuator, "Parameters")
                xs_TotColumns = LightXML.find_element(xs_Actuator, "Tot_Columns")
                xs_IVL = LightXML.find_element(xs_Actuator,"IVL")
                xs_OVL = LightXML.find_element(xs_Actuator,"OVL")
                xs_Columns = LightXML.find_element(xs_Actuator,"Columns")
                xs_Preprocessor = LightXML.find_element(xs_Actuator,"Preprocessor")
                xs_PreprocessorParameters = LightXML.find_element(xs_Actuator,"Preprocessor_Parameters")
                xs_Postprocessor = LightXML.find_element(xs_Actuator,"Postprocessor")
                xs_PostprocessorParameters = LightXML.find_element(xs_Actuator,"Postprocessor_Parameters")
                xs_ColumnList = LightXML.find_element(xs_Actuator,"Column_List")
                xs_HRes = LightXML.find_element(xs_Actuator,"HRes")
                xs_Name = LightXML.find_element(xs_Actuator,"Name")
                
                A = Actuator()
                A.a_type = LightXML.content(xs_ActuatorType)
                A.parameters = eval(parse(LightXML.content(xs_Parameters)))
                A.tot_columns = eval(parse(LightXML.content(xs_TotColumns)))
                A.ivl = eval(parse(LightXML.content(xs_IVL)))
                A.ovl = eval(parse(LightXML.content(xs_OVL)))
                A.columns = eval(parse(LightXML.content(xs_Columns)))
                A.preprocessor = eval(parse(LightXML.content(xs_Preprocessor)))
                A.preprocessor_parameters = eval(parse(LightXML.content(xs_PreprocessorParameters)))
                A.postprocessor = eval(parse(LightXML.content(xs_Postprocessor)))
                A.postprocessor_parameters = eval(parse(LightXML.content(xs_PostprocessorParameters)))
                A.column_list = eval(parse(LightXML.content(xs_ColumnList)))
                A.name = eval(parse(LightXML.content(xs_Name)))
                return A
            end
                
        function read_XML_DRRP(xs_DRRP)
            xs_NeuralSubstrate = LightXML.find_element(xs_DRRP, "Neural_Substrate")
            xs_HiddenLayer = LightXML.find_element(xs_DRRP, "Hidden_Layer")
            xs_OutputWeights = LightXML.find_element(xs_DRRP,"Output_Weights")
            xs_C = LightXML.find_element(xs_DRRP, "C")
            xs_Initialized = LightXML.find_element(xs_DRRP,"Initialization_Flag")
            xs_MinBlockSize = LightXML.find_element(xs_DRRP,"Min_Block_Size")
            xs_PrefBlockSize = LightXML.find_element(xs_DRRP,"Pref_Block_Size")
            xs_LearningType = LightXML.find_element(xs_DRRP,"Learning_Type")
            xs_DRRPType = LightXML.find_element(xs_DRRP,"DRRP_Type")
            xs_M = LightXML.find_element(xs_DRRP,"M")
            xs_Sensors = LightXML.find_element(xs_DRRP,"Sensors")
            xs_Actuators = LightXML.find_element(xs_DRRP,"Actuators")
            xs_Normalizer = LightXML.find_element(xs_DRRP,"Normalizer")
            
            DRRP = DeepRidgeRegressedPredictor(1,0,1,[Sensor()],[Actuator()])
            if size(DRRP.neural_substrate)[1] != 0 
                DRRP.neural_substrate = read_XML_NeuralSubstrate(xs_NeuralSubstrate)
            end
            DRRP.hidden_layer = read_XML_NeuralLayer(xs_HiddenLayer)
            DRRP.output_weights = eval(parse(LightXML.content(xs_OutputWeights)))''
            DRRP.c = eval(parse(LightXML.content(xs_C)))
            DRRP.initialized = eval(parse(LightXML.content(xs_Initialized)))
            DRRP.min_block_size = eval(parse(LightXML.content(xs_MinBlockSize)))
            DRRP.pref_block_size = eval(parse(LightXML.content(xs_PrefBlockSize)))
            DRRP.learning_type = LightXML.content(xs_LearningType)
            DRRP.drrp_type = LightXML.content(xs_DRRPType)
            if size(DRRP.m)[1] != 0
                DRRP.m = eval(parse(LightXML.content(xs_M)))
            end
            DRRP.sensors = read_XML_Sensors(xs_Sensors)
            DRRP.actuators = read_XML_Actuators(xs_Actuators)
            DRRP.normalizer = eval(symbol(LightXML.content(xs_Normalizer)))
            return DRRP
        end
                   
        function read_XML_COMMITTEE(xs_COMMITTEE)t
            C = COMMITTEE()
            xs_AgentList = LightXML.find_element(xs_COMMITTEE,"Agent_List")  
            Agent_List = []
            for xs_Agent in LightXML.child_elements(xs_AgentList)
                Agent = AGENT(1,[1],0,1,[Sensor()],[Actuator()],"drrp",INIT_CONFIG())
                read_XML_Agent_Struct!(Agent,xs_Agent)
                Agent_List = [Agent; Agent_List]
            end
            C.agent_list = Agent_List
            xs_VotingMethod = LightXML.find_element(xs_COMMITTEE,"Voting_Method")
            C.voting_method = eval(parse(LightXML.content(xs_VotingMethod)))
            return C
        end
                       
            function read_XML_NeuralSubstrate(xs_NeuralSubstrate)
                Neural_Substrate = []
                for xs_NeuralLayer in LightXML.child_elements(xs_NeuralSubstrate)
                    Neural_Substrate = [read_XML_NeuralLayer(xs_NeuralLayer);Neural_Substrate]
                end
                Neural_Substrate = reverse(Neural_Substrate)
                return Neural_Substrate
            end
            
            function read_XML_NeuralLayer(xs_NeuralLayer)
                Neural_Layer = []
                for xs_Neuron in LightXML.child_elements(xs_NeuralLayer)
                    Neural_Layer = [read_XML_Neuron(xs_Neuron);Neural_Layer]
                end
                Neural_Layer = reverse(Neural_Layer)
                return Neural_Layer
            end
                        
            function read_XML_Neuron(xs_Neuron)
                xs_WeightVector = LightXML.find_element(xs_Neuron, "Weight_Vector")
                xs_AdvancedWeights = LightXML.find_element(xs_Neuron, "Advanced_Weights")
                xs_Bias = LightXML.find_element(xs_Neuron, "Bias")
                xs_AGRF = LightXML.find_element(xs_Neuron,"AGRF")
                xs_AF = LightXML.find_element(xs_Neuron,"AF")
                xs_NeuronType = LightXML.find_element(xs_Neuron,"Neuron_Type")
                xs_Plasticity = LightXML.find_element(xs_Neuron,"Plasticity")
                xs_Parameters = LightXML.find_element(xs_Neuron,"Parameters")
                xs_Coordinates = LightXML.find_element(xs_Neuron,"Coordinates")
                xs_From = LightXML.find_element(xs_Neuron,"From")
                xs_To = LightXML.find_element(xs_Neuron,"To")
                xs_LastInput = LightXML.find_element(xs_Neuron,"Last_Input")
                
                Neuron = NEURON(1)
                Neuron.weight_vector = eval(parse(LightXML.content(xs_WeightVector)))
                Neuron.advanced_weights =eval(parse(LightXML.content(xs_AdvancedWeights)))
                Neuron.bias = eval(parse(LightXML.content(xs_Bias)))
                Neuron.agrf = eval(symbol(LightXML.content(xs_AGRF)))
                Neuron.af = eval(symbol(LightXML.content(xs_AF)))
                Neuron.neuron_type = LightXML.content(xs_NeuronType)
                Neuron.plasticity = eval(symbol(LightXML.content(xs_Plasticity)))
                Neuron.coordinates = eval(parse(LightXML.content(xs_Coordinates)))
                Neuron.from = eval(parse(LightXML.content(xs_From)))
                Neuron.to = eval(parse(LightXML.content(xs_To)))
                Neuron.last_input = eval(parse(LightXML.content(xs_LastInput)))
                return Neuron
            end
            
function agent_fit!(Agent::AGENT, Percept, Expected)
    NeuralInput = sense(Agent.sensors,Percept)
    NeuralOutput = think!(Agent,NeuralInput,Expected)
    Output = act(NeuralOutput)
	
	return Output
end

    function reformat_percept!(Percept,VL)
        if VL < size(Percept)[2]
            U_Percept = []
            for i in 1:size(Percept)[1]
                for SW_Pos in 1:(size(Percept)[2] - VL + 1)
                    U_Percept = [U_Percept Percept[SW_Pos:SW_Pos+VL-1]]
                end
            end
            Percept = U_Percept'
            println("reformat_percept!:: VL < size(Percept)[2]")
        elseif VL > size(Percept)[2]
            U_Percept = []
            Diff = VL - size(Percept)[2]
            Padding = [-1.0 for i in 1:Diff]
            for i in 1:size(Percept)[1]
                [U_Percept [Percept[i,:],Padding]]
            end
            Percept = U_Percept'
            println("reformat_percept!:: VL > size(Percept)[2]")
        end
    end

    function sense(Sensors::Array{Sensor}, Percept::Array)
        Sensor = Sensors[1]
        HRes = Sensor.hres
        SensorName = Sensor.name
        ColumnList = Sensor.column_list
        if HRes == 1#Row by Row. Whether the row has a single or multiple columns does not matter.
            n_observations = size(Percept)[1]
            NeuralInput = Array(Float64, n_observations, Sensor.ovl)
            for i = 1:n_observations
                NeuralInput[i,:] = SensorName(Percept,i,ColumnList == Colon() ? collect(1:(size(Percept)[2])) : ColumnList,HRes)   
            end
            return NeuralInput
        else#Sliding window, or multiple rows at a time. Whether the row has a single or multiple columns does not matter.
            n_observations = size(Percept)[1]
            NeuralInput = Array(Float64,n_observations-HRes,Sensor.ovl)
            for i = (1+HRes):n_observations
                NeuralInput[i,:] = SensorName(Percept,i,ColumnList == Colon() ? (1:(size(Percept)[2])) : ColumnList,HRes)
            end
            return NeuralInput
        end
    end

    function think!(Agent,NeuralInput,Expected)
	    if Agent.brain_type == "drrp"
		    fit!(Agent.brain, NeuralInput, Expected)
	    end
    end

	function think(Agent,NeuralInput)
	    if Agent.brain_type == "drrp"
		    NeuralOutput = predict(Agent.brain, NeuralInput)
	    end
	end

	function act!(Actuators,NeuralOutput)
	
	end
	
	function act(NeuralOutput)
	    return NeuralOutput
	end
	
function agent_predict(Agent::AGENT,Percept::Array)
    NeuralInput = sense(Agent.sensors,Percept)
    NeuralOutput = think(Agent,NeuralInput)
    Output = act(NeuralOutput)
	return Output
end

function agent_size(Agent::AGENT)
	if Agent.brain_type == "drrp"
		length(Agent.brain.hidden_layer)
	end
end

function weight_norm(Agent::AGENT)
    Elm = Agent.brain
    OutputWeights = Elm.output_weights
    Weights_Norm = norm(abs(OutputWeights))
    return Weights_Norm
end

function load_agent(FileName::ASCIIString)
	Stream = open(FileName,"r")
	Agent = deserialize(Stream)
	close(Stream)
	return Agent
end

function process_data(Agent::AGENT,DataSet)
 	Sensor = Agent.sensors[1]
    HRes = Sensor.hres
    agent_fit!(Agent,SetIn,SetExp)
    Output = agent_predict(Agent,DataSet)'
    return Output
end

function process_data(AgentFileName::ASCIIString, DataSet)
	Agent = load_agent(AgentFileName)
	process_data(Agent,DataSet)
end
	
function stat_analysis(AgentFileName,Tst_In,Tst_exp)
	Agent = load_agent(AgentFileName::ASCIIString)
	Predictions = Output = process_data(Agent,Tst_In)
	
	Tot_Classes = count_classes(Tst_In[:,1])
	Class_Predictions = vec(sparse_bin_to_int(Predictions,Tot_Classes))
    Class_Expected = vec(sparse_bin_to_int(Tst_Exp,Tot_Classes))#[HRes+1:end,1]
    TstAUC = auc(Class_Expected .- 1, Class_Predictions .- 1, 200)
    SparseBin_Predictions = int_to_sparse_bin2(Class_Predictions,Tot_Classes,-1)
    Prediction_ClassDistribution = fill(0,Tot_Classes)
   	Expected_ClassDistribution = fill(0,Tot_Classes)
    get_ClassDistribution!(Class_Predictions,Prediction_ClassDistribution,-1)
    get_ClassDistribution!(Class_Expected,Expected_ClassDistribution,-1)
    TstAvgErr = avgerr(Tst_Exp[1:end,:],SparseBin_Predictions,SparseBin_Expected,Sensors)
    TstRMS = rms(Tst_Exp[1:end,:],SparseBin_Predictions,SparseBin_Expected,Sensors)
    TstAccuracy = accuracy(SparseBin_Predictions,SparseBin_Expected)
    
    println("###################### CLASSIFICATION TEST RESULTS vvvvSTARTvvvvvvvvvvvvvvvvvv")
    println("TstAccuracy: $TstAccuracy, TstAUC:$TstAUC, TstRMS: $TstRMS, TstAvgErr: $TstAvgErr")
    println("#######################CLASSIFICATION TEST RESULTS ^^^^END^^^^^^^^^^^^^^^^^^^^")
    return (TstAccuracy, TstRMS, TstAvgErr, TstAUC)
end

function agents_SA(AgentIdsName,Tst_In,Tst_Exp)
	AgentIdsStream = Stream = open(FileName,"r")
	for AgentId in eachline(AgentIdsStream)#[]
		println("AgentName:$AgentId")
		AgentFileName = string("./Input_Files/agent_TstBest_ValFitness_",AgentId,".dat")
		(TstAccuracy,TstRMS,TstAvgErr,TstAUC) = stat_analysis(AgentFileName,Tst_In,Tst_Exp)
		Println("TstAccuracy:$TstAccuracy, TstRMS:$TstRMS, TstAvgErr:$TstAvgErr, TstAUC:$TstAUC")
	end
end

function rescale(A)
    rescale(A,1)
end

function rescale(A,ColumnIndex::Int)
    rescale(A,maximum(A[:,ColumnIndex]),minimum(A[:,ColumnIndex]),ColumnIndex)
end

function rescale(A,Max,Min,ColumnIndex)
	B = deepcopy(A[:,ColumnIndex])
	for i in 1:size(A)[1]
	    B[i] = rescale(A[i,ColumnIndex],Max,Min)
	end
	return B
end

function rescale(Val::Any,Max,Min)#Nm = (Y*2 - (Max + Min))/(Max-Min)
	if Max == Min
	    return 0
	else
	    return (Val*2 - (Max+Min))/(Max-Min)
	end
end

function neuron_compute(N, Input)
	if N.af == mismatch_kernel
	    mismatch_kernel(N,Input)
	elseif N.af == ngram_kernel
		ngram_kernel(N,Input)
	else
		Agrf = N.agrf
		Af = N.af
		Acc = Agrf(N.weight_vector, N.bias, Input)
		Output = Af(Acc)
		return Output
	end
end

function ngram_kernel(N::NEURON,Input)
	Output = ngram_kernel(N.weight_vector,Input)
	return Output
end

function mismatch_kernel(N::NEURON,Input)
	Output = mismatch_kernel(N.weight_vector,Input)
	return Output
end


#################################################################################### AGGREGATION FUNCTIONS #################

function af_dot(Weights,Bias,Input)
	return dot(Weights,Input) + Bias
end

#################################################################################### ACTIVATION FUNCTIONS #################

function sigmoid(Aggregator,Weights,Input,Bias)
	return sigmoid(Aggregator(Weights,Input) + Bias)
end

	function sigmoid(x)
		return 1.0 ./ (1.0 + exp(-x))
	end

	function ngram_kernel(NGram,Sequence)
		Acc = 0
		for N in 1:length(Sequence)+1-length(NGram)
			Substring = (Sequence[N:N+length(NGram)-1])
			if NGram == Substring
				Acc += 1
			end
		end
		return Acc
	end
	
	
	function mismatch_kernel(NGram,Sequence)
	    Acc = 0
	    ErrAcc = 0
	    NGram_Size = length(NGram)
	    ErrAllowed = round(sqrt(NGram_Size))
		for N in 1:length(Sequence)+1-NGram_Size
			Substring = (Sequence[N:N+length(NGram)-1])
			ErrAcc = 0
			for i in 1:NGram_Size
			    if NGram[i] != Substring[i]
			        ErrAcc += 1
			    end
			end
			if ErrAcc <= ErrAllowed
				Acc += 1
			end
		end
		return Acc
	end

#################################################################################### NORMALIZATION FUNCTIONS #################
function none(DataSet)
    return DataSet
end

function proportionalize!(DataSet)
	DataSet = DataSet ./ sum(DataSet, 2)
end

function proportionalize(DataSet)
	return DataSet ./ sum(DataSet, 2)
end

function proportionalize_row!(DataSet,Row_Index)
	Acc = sum(DataSet[Row_Index,:])
	if Acc != 0
		DataSet[Row_Index,:] = (DataSet[Row_Index,:]/Acc)
	end
end

function normalize_row!(DataSet,Row_Index)
	Acc = 0
	for i in [1:size(DataSet)[2]]
		Acc = Acc + DataSet[Row_Index,i]^2
	end
	RMS = sqrt(Acc)
	if RMS != 0
		DataSet[Row_Index,:] = DataSet[Row_Index,:]/RMS
	end
end
	
function find_activations(Neuron_Layer::Array{NEURON}, Sensors::Array{Sensor}, NeuralInput::Array)
        Sensor = Sensors[1]
        HRes = Sensor.hres
        SensorName = Sensor.name
        ColumnList = Sensor.column_list
	    if HRes == 1#Row by Row. Whether the row has a single or multiple columns does not matter.
            n_observations = size(NeuralInput)[1]
            act_matrix = Array(Float64,length(Neuron_Layer), n_observations)
	        for i = 1:n_observations
		        for k in 1:length(Neuron_Layer)
		        	act_matrix[k,i] = neuron_compute(Neuron_Layer[k],NeuralInput[i,:])
		        end
	        end
	    else#Sliding window, or multiple rows at a time. Whether the row has a single or multiple columns does not matter.
	    	n_observations = size(NeuralInput)[1]
	        act_matrix = Array(Float64,length(Neuron_Layer), n_observations-HRes)
	        for i = (1+HRes):n_observations
	            for k in 1:layer.n_hidden_neurons
		            act_matrix[k, i-HRes] = neuron_compute(Neuron_Layer[k],NeuralInput[i,:])
		        end 
	        end
	    end
	act_matrix
end
####################################### DRREP SIGNAL PROCESSING #######################################
function predict(Elm::DeepRidgeRegressedPredictor, NeuralInput::Vector)
	if length(Elm.hidden_layer) == 0
		return SetIn
	else
		Sensors = Elm.sensors
		Activation_Matrix = find_activations(Elm.hidden_layer, Sensors, NeuralInput)
		Normalizer = Elm.normalizer
		Normalizer(Activation_Matrix)
		NeuralOutput = Elm.output_weights * Activation_Matrix
		return NeuralOutput
	end
end

function predict(Elm::DeepRidgeRegressedPredictor, NeuralInput::Matrix)
	if length(Elm.hidden_layer) == 0
		return SetIn
	else
		Sensors = Elm.sensors
		Activation_Matrix = find_activations(Elm.hidden_layer, Sensors, NeuralInput)
		Normalizer = Elm.normalizer
		Normalizer(Activation_Matrix)
		NeuralOutput = Elm.output_weights * Activation_Matrix
		return NeuralOutput
	end
end

end

if true
    using LightXML
    using mymod
    mymod.main()
end
